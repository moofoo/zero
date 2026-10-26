// A headless duel simulator (bot-ai ticket 41), first checkpoint: two zero game clients in one process, on a virtual
// clock, with an in-memory relay in place of the server. Each client simulates its own ship and takes its own damage,
// as in the real game; the relay passes positions (with any weapon fired) between them and turns deaths into kills.
// Each client's arena setup is replayed from packets captured from a real game (zero's ZERO_CAPTURE).
//
// duelsim <capture A> <capture B> <seconds>   (run from a bots folder holding zones/Chaos/<map>.lvl)
// duelsim player <capture> <input log> <out> [combat log]   (a player's recording as fair observations; ticket 46)
#include <zero/game/Clock.h>
#include <zero/game/Settings.h>
#include <zero/game/Game.h>
#include <zero/game/InputState.h>
#include <zero/game/Logger.h>
#include <zero/game/Memory.h>
#include <zero/game/WorkQueue.h>
#include <zero/game/net/Connection.h>
#include <zero/game/GameEvent.h>
#include <zero/game/InputLog.h>
#include <zero/game/ShipController.h>
#include <zero/Args.h>
#include <zero/BotController.h>
#include <zero/Config.h>
#include <zero/RenderContext.h>
#include <zero/ZeroBot.h>
#include <zero/commands/CommandSystem.h>
#include <zero/policy/DuelPolicy.h>
#include <zero/zones/ZoneController.h>
#include <zero/zones/chaos/Chaos.h>
#include <zero/Seed.h>
#include <zero/policy/FairObservation.h>

#include <algorithm>
#include <cmath>
#include <map>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

using namespace zero;

namespace zero {
// Globals the client expects with the bot's main: the server name, whose folder (zones/<name>) holds the cached map,
// and the security service address. ZeroBot.cpp, built in for the bots (ticket 45), has the player name and settings.
extern const char* kPlayerName;
extern const char* kPlayerPassword;
const char* kServerName = "Chaos";
const char* kSecurityServiceIp = "127.0.0.1";
}  // namespace zero

namespace {

struct Packet {
  u8 direction;  // 0 server to client, 1 client to server
  u32 tick;
  std::vector<u8> data;
};

std::vector<Packet> ReadCapture(const char* path) {
  std::vector<Packet> packets;
  FILE* f = fopen(path, "rb");
  if (!f) return packets;
  while (true) {
    Packet p;
    u32 size;
    if (fread(&p.direction, 1, 1, f) != 1 || fread(&p.tick, 4, 1, f) != 1 || fread(&size, 4, 1, f) != 1) break;
    p.data.resize(size);
    if (fread(p.data.data(), 1, size, f) != size) break;
    packets.push_back(std::move(p));
  }
  fclose(f);
  return packets;
}

struct Side;

// The relay: what the server would send each client.
struct Relay {
  Side* sides[2] = {};
  std::deque<std::pair<int, std::vector<u8>>> outbox;  // (to side, packet)
  void FromClient(int from, u8* data, size_t size);
};

struct Side {
  int index = 0;
  // The side's event scope (Event.h): events from its client reach only its own handlers, as if it ran alone.
  unsigned int scope = 0;
  Relay* relay = nullptr;
  u16 pid = 0xFFFF;
  MemoryArena perm{nullptr, 0}, trans{nullptr, 0}, work{nullptr, 0};
  WorkQueue* queue = nullptr;
  Worker* worker = nullptr;
  Game* game = nullptr;
  InputState input;
  ArenaSnapshot scratch;  // the transient arena after setup: reverted to after every update
  // Later security packets (door and prize seeds), delivered at their own time in a replay: (server tick, packet).
  std::vector<std::pair<u32, std::vector<u8>>> deferred;
  size_t next_deferred = 0;
  // A zero Chaos bot flying this side (ticket 45), made before the login replays so it hears the arena and map
  // events: its bot config file and arguments (as zero's command line: --skill veteran --aggression standard ...).
  std::string bot_config, bot_args;
  ZeroBot* bot = nullptr;
  std::unique_ptr<ZoneController> controller;
  std::vector<std::string> bot_argv;
  u32 bot_seed = 0;  // the next seed for the bot's random generators (ChaosRandom.h)
};

// Runs `work` in the side's event scope with the bot's seeds, then back to scope 0.
template <typename F>
void InScope(Side& side, F&& work) {
  g_EventScope = side.scope;
  if (side.bot) g_Seed = side.bot_seed;
  work();
  if (side.bot) side.bot_seed = *g_Seed;
  g_Seed.reset();
  g_EventScope = 0;
}

// Makes the side's bot around its game, as ZeroBot::JoinZone does without the connection and window.
bool MakeBot(Side& side) {
  ZeroBot* bot = new ZeroBot();
  bot->game = side.game;
  bot->config = Config::Load(side.bot_config.c_str());
  if (!bot->config) {
    fprintf(stderr, "duelsim: can't load the bot config %s\n", side.bot_config.c_str());
    return false;
  }
  side.bot_argv = {"zero"};
  size_t start = 0;
  while (start < side.bot_args.size()) {
    size_t end = side.bot_args.find(' ', start);
    if (end == std::string::npos) end = side.bot_args.size();
    if (end > start) side.bot_argv.push_back(side.bot_args.substr(start, end - start));
    start = end + 1;
  }
  std::vector<char*> argv;
  for (auto& a : side.bot_argv) argv.push_back(a.data());
  bot->args = std::make_unique<ArgParser>((int)argv.size(), argv.data());
  strcpy(bot->name, "SimBot");
  bot->bot_controller = new BotController(*side.game);
  bot->commands = new CommandSystem(*bot, side.game->dispatcher);
  bot->server_info.name = "Chaos";
  bot->server_info.zone = Zone::Chaos;
  bot->execute_ctx.bot = bot;  // zero sets it every update, before the arena's packets arrive
  bot->execute_ctx.dt = 0.01f;
  side.controller = chaos::CreateChaosController();
  side.bot = bot;
  Event::Dispatch(ZeroBot::JoinRequestEvent(*bot, bot->server_info));
  return true;
}

// The bot's think for this tick (ZeroBot::Update's first half): its tree sets the side's input.
void BotThink(Side& side) {
  ZeroBot& bot = *side.bot;
  Game& game = *side.game;
  if (bot.bot_controller->actuator.enabled) {
    side.input.Clear();
  } else {
    side.input.ClearWeapons();
  }
  if (game.connection.login_state != Connection::LoginState::Complete) return;
  bot.execute_ctx.bot = &bot;
  bot.execute_ctx.dt = 0.01f;
  RenderContext rc(&game.camera, &game.ui_camera, &game.sprite_renderer);
  bot.bot_controller->Update(rc, side.input, bot.execute_ctx);
}

void Loopback(void* user, u8* data, size_t size) {
  Side* side = (Side*)user;
  side->relay->FromClient(side->index, data, size);
}

void Write16(std::vector<u8>& v, u16 x) {
  v.push_back((u8)(x & 0xFF));
  v.push_back((u8)(x >> 8));
}

u16 Read16(const u8* p) { return (u16)(p[0] | (p[1] << 8)); }
u32 Read32(const u8* p) { return (u32)(p[0] | (p[1] << 8) | (p[2] << 16) | ((u32)p[3] << 24)); }

void Relay::FromClient(int from, u8* data, size_t size) {
  Side& me = *sides[from];
  int to = 1 - from;
  if (size == 0) return;
  if (data[0] == 0x00 && size >= 2) {
    // Core: a reliable message is unwrapped and acknowledged; the rest (time sync and so on) isn't needed.
    if (data[1] == 0x03 && size >= 6) {
      u32 id = Read32(data + 2);
      std::vector<u8> ack = {0x00, 0x04, data[2], data[3], data[4], data[5]};
      outbox.push_back({from, ack});
      FromClient(from, data + 6, size - 6);
    } else if (data[1] == 0x0E) {
      // A cluster: [size u8][packet]...
      size_t i = 2;
      while (i < size) {
        size_t n = data[i];
        if (i + 1 + n > size) break;
        FromClient(from, data + i + 1, n);
        i += 1 + n;
      }
    }
    return;
  }
  if (data[0] == 0x03 && size >= 22) {
    // Position (C2S 0x03): type, direction, timestamp u32, vel_x, y, checksum, togglables, x, vel_y, bounty, energy,
    // weapon [extra]. Sent on as a large position (S2C 0x05) to the other client.
    const u8* p = data;
    std::vector<u8> out;
    out.push_back(0x05);
    out.push_back(p[1]);                 // direction
    Write16(out, Read16(p + 2));         // timestamp, low 16 bits
    Write16(out, Read16(p + 12));        // x
    Write16(out, Read16(p + 14));        // vel_y
    Write16(out, me.pid);                // pid
    Write16(out, Read16(p + 6));         // vel_x
    out.push_back(p[10]);                // checksum
    out.push_back(p[11]);                // togglables
    out.push_back(0);                    // ping
    Write16(out, Read16(p + 8));         // y
    Write16(out, Read16(p + 16));        // bounty
    Write16(out, Read16(p + 20));        // weapon
    if (size >= 32) {
      Write16(out, Read16(p + 18));      // energy
      Write16(out, 0);                   // s2c latency
      Write16(out, Read16(p + 26));      // timer
      for (int k = 0; k < 4; ++k) out.push_back(p[28 + k]);  // items
    }
    outbox.push_back({to, out});
    return;
  }
  if (data[0] == 0x05 && size >= 5) {
    // Death (C2S 0x05): killer u16, bounty u16. Every client gets a kill (S2C 0x06).
    std::vector<u8> out = {0x06, 0x00};
    Write16(out, Read16(data + 1));  // killer
    Write16(out, me.pid);            // killed
    Write16(out, Read16(data + 3));  // bounty
    Write16(out, 0);                 // flags
    outbox.push_back({0, out});
    outbox.push_back({1, out});
    return;
  }
}

// With `until`, only packets captured up to that server tick are replayed (a player's recording: the rest come in
// their own time); `*replayed` is then how many of the capture's packets that was.
bool SetupSide(Side& side, const std::vector<Packet>& capture, Tick start, s32 offset = 0, bool defer = false,
               u32 until = 0, size_t* replayed = nullptr) {
  constexpr size_t kPermanentSize = Megabytes(64), kTransientSize = Megabytes(32), kWorkSize = Megabytes(4);
  side.perm = MemoryArena((u8*)malloc(kPermanentSize), kPermanentSize);
  side.trans = MemoryArena((u8*)malloc(kTransientSize), kTransientSize);
  side.work = MemoryArena((u8*)malloc(kWorkSize), kWorkSize);
  side.queue = new WorkQueue(side.work);
  side.worker = new Worker(*side.queue);
  side.worker->Launch();
  g_EventScope = side.scope;
  if (!side.bot_config.empty()) g_Seed = side.bot_seed;
  side.game = memory_arena_construct_type(&side.perm, Game, side.perm, side.trans, *side.queue, 1920, 1080);
  if (side.game->Initialize(side.input) == GameInitializeResult::Failure) return false;
  Connection& c = side.game->connection;
  c.loopback = Loopback;
  c.loopback_user = &side;
  c.connected = true;
  if (!side.bot_config.empty() && !MakeBot(side)) return false;
  // Replay the arena setup: every server packet but positions, kills, chat and keep-alives.
  bool seen_security = false;
  size_t count = 0;
  for (const Packet& p : capture) {
    if (until && TICK_GT((u32)((s32)p.tick + offset) & 0x7fffffff, until)) break;
    ++count;
    if (p.direction != 0 || p.data.empty()) continue;
    u8 type = p.data[0];
    if (type == 0x00 || type == 0x05 || type == 0x28 || type == 0x06 || type == 0x07 || type == 0x27) continue;
    if (type == 0x01 && p.data.size() >= 3) side.pid = Read16(p.data.data() + 1);
    // A security packet after the first re-seeds the doors from its own timestamp: replaying it now would freeze them.
    if (type == 0x18 && defer && seen_security) {
      side.deferred.push_back({(u32)((s32)p.tick + offset) & 0x7fffffff, p.data});
      continue;
    }
    if (type == 0x18) seen_security = true;
    SetVirtualTick(start);
    std::vector<u8> copy = p.data;
    c.ProcessPacket(copy.data(), copy.size());
  }
  if (replayed) *replayed = count;
  c.time_diff = 0;
  side.scratch = side.trans.GetSnapshot();
  if (side.bot) side.bot_seed = *g_Seed;
  g_Seed.reset();
  g_EventScope = 0;
  return side.pid != 0xFFFF;
}


// --- Replay: drive both clients with a real game's logged inputs and compare -------------------------------------

struct LoggedUpdate {
  u32 tick;  // server tick
  u32 actions;
  std::vector<u8> one_shots;  // one-shot actions just before this update
  float x, y, vx, vy, orientation, energy, enter_delay;
  u8 ship;
  u32 fields[kShipFields];  // the ship's upgrades (InputLog.h)
  float dt = 0.01f;  // the update's time: a tick in zero's logs, a frame's in nullspace's (record 3)
};

std::vector<LoggedUpdate> ReadInputLog(const char* path, s32 offset) {
  std::vector<LoggedUpdate> updates;
  std::vector<u8> pending;
  FILE* f = fopen(path, "rb");
  if (!f) return updates;
  while (true) {
    u8 type;
    u32 tick;
    if (fread(&type, 1, 1, f) != 1 || fread(&tick, 4, 1, f) != 1) break;
    if (type == 2) {
      u8 action;
      if (fread(&action, 1, 1, f) != 1) break;
      pending.push_back(action);
    } else if (type == 1 || type == 3) {  // 3: nullspace's frame update, with its dt after
      LoggedUpdate u;
      float v[6];
      if (fread(&u.actions, 4, 1, f) != 1 || fread(v, 4, 6, f) != 6 || fread(&u.ship, 1, 1, f) != 1 ||
          fread(&u.enter_delay, 4, 1, f) != 1 || fread(u.fields, 4, kShipFields, f) != (size_t)kShipFields)
        break;
      if (type == 3 && fread(&u.dt, 4, 1, f) != 1) break;
      u.tick = (u32)((s32)tick + offset) & 0x7fffffff;
      u.x = v[0], u.y = v[1], u.vx = v[2], u.vy = v[3], u.orientation = v[4], u.energy = v[5];
      u.one_shots.swap(pending);
      updates.push_back(u);
    } else {
      break;
    }
  }
  fclose(f);
  return updates;
}

// A client's server-time offset: from its captured position packets, the server timestamp it sent less the local tick
// it was sent at.
// The median over every position packet: the first ones can go out before the client has synced its clock.
s32 ServerOffset(const std::vector<Packet>& capture) {
  std::vector<s32> offsets;
  for (const Packet& p : capture) {
    if (p.direction == 1 && !p.data.empty() && p.data[0] == 0x03 && p.data.size() >= 6) {
      offsets.push_back((s32)Read32(p.data.data() + 2) - (s32)p.tick);
    }
  }
  if (offsets.empty()) return 0;
  std::sort(offsets.begin(), offsets.end());
  if (getenv("SIM_DEBUG")) fprintf(stderr, "offsets: first %d, median %d, last %d\n", offsets.front(), offsets[offsets.size() / 2], offsets.back());
  return offsets[offsets.size() / 2];
}

struct HitRecord {
  u32 tick;
  int victim;  // side
  int weapon;  // class: 0 bullet (either kind), 1 bomb (bomb, proximity bomb, thor), 2 anything else
};

// As the combat log names weapons: shrapnel is a bullet carrying its bomb's shrapnel count, a mine a bomb laid.

std::vector<HitRecord>* sim_hits = nullptr;
Side* replay_sides = nullptr;

std::vector<std::pair<u32, int>> sim_fires[2];  // (tick, weapon type) per side

struct FireWatcher : EventHandler<WeaponFireEvent> {
  void HandleEvent(const WeaponFireEvent& event) override {
    if (!replay_sides) return;
    for (int i = 0; i < 2; ++i) {
      if (replay_sides[i].game->player_manager.GetSelf() == &event.player) {
        sim_fires[i].push_back({GetCurrentTick(), (int)event.data.type});
        return;
      }
    }
  }
};

struct HitWatcher : EventHandler<WeaponSelfDamageEvent> {
  void HandleEvent(const WeaponSelfDamageEvent& event) override {
    if (!sim_hits || !replay_sides) return;
    for (int i = 0; i < 2; ++i) {
      PlayerManager& pm = replay_sides[i].game->player_manager;
      if (&event.shooter >= pm.players && &event.shooter < pm.players + pm.player_count) {
        sim_hits->push_back({GetCurrentTick(), i, WeaponClass(event.weapon.data)});
        return;
      }
    }
  }
};

// Real hits from a bot's events log (its own "hit" events: hits taken, in server ticks).
std::vector<HitRecord> ReadRealHits(const char* path, int victim) {
  std::vector<HitRecord> hits;
  FILE* f = fopen(path, "r");
  if (!f) return hits;
  char line[4096];
  while (fgets(line, sizeof(line), f)) {
    if (!strstr(line, "\"e\":\"hit\"")) continue;
    const char* t = strstr(line, "\"tick\":");
    const char* w = strstr(line, "\"weapon\":\"");
    if (!t || !w) continue;
    u32 tick = (u32)strtoul(t + 7, nullptr, 10);
    w += 10;
    int weapon = strncmp(w, "bullet", 6) == 0 || strncmp(w, "bounce", 6) == 0 ? 0
                 : strncmp(w, "bomb", 4) == 0 || strncmp(w, "prox", 4) == 0 || strncmp(w, "thor", 4) == 0 ? 1
                                                                                               : 2;
    hits.push_back({tick, victim, weapon});
  }
  fclose(f);
  return hits;
}

int Replay(char** argv) {
  // replay <capA> <inpA> <eventsA> <capB> <inpB> <eventsB>
  std::vector<Packet> caps[2] = {ReadCapture(argv[2]), ReadCapture(argv[5])};
  s32 offsets[2] = {ServerOffset(caps[0]), ServerOffset(caps[1])};
  std::vector<LoggedUpdate> logs[2] = {ReadInputLog(argv[3], offsets[0]), ReadInputLog(argv[6], offsets[1])};
  if (logs[0].empty() || logs[1].empty()) {
    fprintf(stderr, "can't read the input logs\n");
    return 1;
  }
  kPlayerName = "sim";
  u32 start = std::min(logs[0].front().tick, logs[1].front().tick);
  if (getenv("SIM_DEBUG")) {
    for (int k = 0; k < 2; ++k) {
      u32 alive = 0;
      for (auto& u : logs[k]) if (u.ship < 8 && u.enter_delay <= 0.0f) { alive = u.tick; break; }
      fprintf(stderr, "side %d: log from %u, first alive %u\n", k, logs[k].front().tick, alive);
    }
  }
  SetVirtualTick(start);
  Relay relay;
  static Side sides[2];
  for (int i = 0; i < 2; ++i) {
    sides[i].index = i;
    sides[i].relay = &relay;
    relay.sides[i] = &sides[i];
  }
  if (!SetupSide(sides[0], caps[0], start, offsets[0], true) || !SetupSide(sides[1], caps[1], start, offsets[1], true)) {
    fprintf(stderr, "setup failed\n");
    return 1;
  }
  relay.outbox.clear();
  std::vector<HitRecord> hits;
  sim_hits = &hits;
  replay_sides = sides;
  HitWatcher watcher;
  FireWatcher fire_watcher;

  std::vector<HitRecord> real_hits = ReadRealHits(argv[4], 0), real_hits_b = ReadRealHits(argv[7], 1);
  real_hits.insert(real_hits.end(), real_hits_b.begin(), real_hits_b.end());
  std::sort(real_hits.begin(), real_hits.end(), [](const HitRecord& x, const HitRecord& y) { return x.tick < y.tick; });
  size_t probe = 0;
  // Errors by time since the last re-sync (start or respawn), per side.
  std::map<int, std::vector<float>> position_error, energy_error;  // keyed by seconds since re-sync (1, 5)
  size_t next[2] = {0, 0};
  u32 synced_at[2] = {0, 0};
  bool synced[2] = {false, false};
  bool was_dead[2] = {true, true};
  int sim_deaths[2] = {0, 0}, real_deaths[2] = {0, 0};
  int upgrade_syncs[2] = {0, 0};
  int teleports[2] = {0, 0};
  bool sim_dead[2] = {false, false};
  float worst5[2] = {0, 0};
  while (next[0] < logs[0].size() || next[1] < logs[1].size()) {
    int i = next[1] >= logs[1].size() || (next[0] < logs[0].size() && logs[0][next[0]].tick <= logs[1][next[1]].tick)
                ? 0 : 1;
    const LoggedUpdate& u = logs[i][next[i]++];
    SetVirtualTick(u.tick);
    Side& side = sides[i];
    Player* self = side.game->player_manager.GetSelf();
    if (!self) continue;
    while (side.next_deferred < side.deferred.size() && side.deferred[side.next_deferred].first <= u.tick) {
      auto& pkt = side.deferred[side.next_deferred++].second;
      side.game->connection.ProcessPacket(pkt.data(), pkt.size());
    }
    for (u8 a : u.one_shots) {
      if (side.input.action_callback) side.input.action_callback(side.input.user, (InputAction)a);
    }
    side.input.actions = u.actions;
    // The ship's upgrades come from the arena's prize rolls at each spawn (randomness, not physics): kept as logged.
    {
      u32 now_fields[kShipFields];
      ShipFields(side.game->ship_controller.ship, now_fields);
      if (memcmp(now_fields, u.fields, sizeof(now_fields)) != 0 && u.ship < 8) {
        if (getenv("SIM_DEBUG") && upgrade_syncs[i] < 6) {
          fprintf(stderr, "sync side %d at %u:", i, u.tick);
          for (int k = 0; k < kShipFields; ++k)
            if (now_fields[k] != u.fields[k]) fprintf(stderr, " [%d] %u->%u", k, now_fields[k], u.fields[k]);
          fprintf(stderr, "\n");
        }
        SetShipFields(side.game->ship_controller.ship, u.fields);
        ++upgrade_syncs[i];
      }
    }
    side.game->Update(side.input, 0.01f);
    side.game->player_manager.kdtree = nullptr;
    side.trans.Revert(side.scratch);
    while (!relay.outbox.empty()) {
      auto [to, pkt] = relay.outbox.front();
      relay.outbox.pop_front();
      sides[to].game->connection.ProcessPacket(pkt.data(), pkt.size());
    }
    if (getenv("SIM_DEBUG") && u.tick % 500 == 0) {
      WeaponManager& wm = side.game->weapon_manager;
      int own = 0, other = 0;
      for (size_t k = 0; k < wm.weapon_count; ++k) (wm.weapons[k].player_id == side.pid ? own : other)++;
      Player* them = side.game->player_manager.GetPlayerById(sides[1 - i].pid);
      // The incoming weapon nearest this ship, and the shooter's own copy nearest to it.
      WeaponManager& theirs = sides[1 - i].game->weapon_manager;
      float best = 1e9f, match = 1e9f;
      Vector2f bp;
      for (size_t k = 0; k < wm.weapon_count; ++k) {
        if (wm.weapons[k].player_id == side.pid) continue;
        float d = wm.weapons[k].position.Distance(self->position);
        if (d < best) best = d, bp = wm.weapons[k].position;
      }
      for (size_t k = 0; k < theirs.weapon_count; ++k) match = std::min(match, theirs.weapons[k].position.Distance(bp));
      fprintf(stderr, "dbg   freqs: self %d, other %d\n", self->frequency, them ? them->frequency : -1);
      if (best < 1e9f) fprintf(stderr, "dbg   nearest incoming %.1f tiles from self; shooter's nearest own weapon %.1f tiles from it\n", best, match);
      fprintf(stderr, "dbg t=%u side %d: weapons own %d other %d | other at (%.1f,%.1f) ship %d delay %.2f | connection weapons received %u\n",
              u.tick, i, own, other, them ? them->position.x : -1.f, them ? them->position.y : -1.f, them ? them->ship : -1,
              them ? them->enter_delay : -1.f, side.game->connection.weapons_received);
    }
    if (getenv("SIM_DEBUG") && probe < 10 && probe < real_hits.size() && real_hits[probe].victim == i &&
        u.tick >= real_hits[probe].tick) {
      WeaponManager& wm = side.game->weapon_manager;
      float best = 1e9f;
      for (size_t k = 0; k < wm.weapon_count; ++k) {
        if (wm.weapons[k].player_id == side.pid) continue;
        best = std::min(best, wm.weapons[k].position.Distance(self->position));
      }
      fprintf(stderr, "probe real hit %zu at %u (weapon %d) on side %d: nearest incoming %.2f tiles; self vs logged %.2f tiles; energy sim %.0f real %.0f\n",
              probe, real_hits[probe].tick, real_hits[probe].weapon, i, best,
              std::hypot(self->position.x - u.x, self->position.y - u.y), self->energy, u.energy);
      ++probe;
    }
    while (probe < real_hits.size() && real_hits[probe].tick < u.tick && real_hits[probe].victim != i && probe < 10 && false) ++probe;
    if (getenv("SIM_DEBUG")) {
      static int overlaps[2] = {0, 0};
      static u32 last_overlap[2] = {0, 0};
      WeaponManager& wm = side.game->weapon_manager;
      for (size_t k = 0; k < wm.weapon_count; ++k) {
        if (wm.weapons[k].player_id == side.pid) continue;
        if (wm.weapons[k].position.Distance(self->position) < 0.9f && u.tick - last_overlap[i] > 5) {
          ++overlaps[i];
          last_overlap[i] = u.tick;
          if (overlaps[i] <= 3)
            fprintf(stderr, "overlap side %d at %u: weapon type %d, self ship %d delay %.2f freq %d, weapon freq %d, kdtree %p\n", i,
                    u.tick, (int)wm.weapons[k].data.type, self->ship, self->enter_delay, self->frequency,
                    wm.weapons[k].frequency, (void*)side.game->player_manager.kdtree);
        }
      }
      if (next[0] >= logs[0].size() - 1 && next[1] >= logs[1].size() - 1)
        fprintf(stderr, "overlaps: side 0 %d, side 1 %d\n", overlaps[0], overlaps[1]);
    }
    bool real_dead = u.ship >= 8 || u.enter_delay > 0.0f;
    if (real_dead && !was_dead[i] && synced[i]) {
      ++real_deaths[i];
      if (getenv("SIM_DEBUG")) fprintf(stderr, "real death side %d at %u\n", i, u.tick);
    }
    bool now_sim_dead = self->ship >= 8 || self->enter_delay > 0.0f;
    if (now_sim_dead && !sim_dead[i] && synced[i]) {
      ++sim_deaths[i];
      if (getenv("SIM_DEBUG")) fprintf(stderr, "sim death side %d at %u (energy real %.0f)\n", i, u.tick, u.energy);
    }
    sim_dead[i] = now_sim_dead;
    // Re-sync at the start, whenever the real ship comes back from a death, and when it jumps (a warp to a random spot,
    // which is the arena's randomness, not physics): put the simulated ship where it was.
    bool jumped = false;
    if (next[i] >= 2) {
      const LoggedUpdate& before = logs[i][next[i] - 2];
      jumped = !real_dead && before.ship < 8 && before.enter_delay <= 0.0f && std::hypot(u.x - before.x, u.y - before.y) > 3.0f;
    }
    if (jumped) ++teleports[i];
    if (!real_dead && (was_dead[i] || !synced[i] || jumped)) {
      self->position = Vector2f(u.x, u.y);
      self->velocity = Vector2f(u.vx, u.vy);
      self->orientation = u.orientation;
      self->energy = u.energy;
      self->enter_delay = 0.0f;
      side.game->ship_controller.ship.energy = side.game->ship_controller.ship.energy;
      synced[i] = true;
      synced_at[i] = u.tick;
    }
    was_dead[i] = real_dead;
    if (!synced[i] || real_dead || now_sim_dead) continue;
    s32 since = (s32)(u.tick - synced_at[i]);
    {
      static bool reported[2] = {false, false};
      float err = std::hypot(self->position.x - u.x, self->position.y - u.y);
      if (err > 0.5f && !reported[i]) {
        reported[i] = true;
        auto& map = side.game->GetMap();
        int doors = 0;
        for (int dy = -3; dy <= 3; ++dy)
          for (int dx = -3; dx <= 3; ++dx) {
            TileId id = map.GetTileId((u16)(u.x + dx), (u16)(u.y + dy));
            if (id >= 162 && id <= 169) ++doors;
          }
        fprintf(stderr, "first divergence side %d at %u (%.1f s after re-sync): sim (%.2f, %.2f) v (%.2f, %.2f) | real (%.2f, %.2f) v (%.2f, %.2f) | energy sim %.0f real %.0f | door tiles near %d\n",
                i, u.tick, since / 100.0f, self->position.x, self->position.y, self->velocity.x, self->velocity.y, u.x, u.y,
                u.vx, u.vy, self->energy, u.energy, doors);
      }
    }
    for (int mark : {100, 500, 1000, 3000}) {
      if (since == mark) {
        float err = std::hypot(self->position.x - u.x, self->position.y - u.y);
        position_error[mark].push_back(err);
        energy_error[mark].push_back(std::fabs(self->energy - u.energy));
        if (mark == 500) worst5[i] = std::max(worst5[i], err);
      }
    }
  }
  sim_hits = nullptr;

  auto median = [](std::vector<float> v) {
    if (v.empty()) return -1.0f;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  printf("replayed %zu and %zu updates\n", logs[0].size(), logs[1].size());
  for (auto& [mark, errs] : position_error) {
    printf("  %4.0f s after a re-sync: position error median %.2f tiles, max %.2f (n %zu); energy error median %.0f\n",
           mark / 100.0f, median(errs), *std::max_element(errs.begin(), errs.end()), errs.size(),
           median(energy_error[mark]));
  }
  for (int k = 0; k < 2; ++k) {
    int bullets = 0, bombs = 0;
    for (auto& f : sim_fires[k]) (f.second == (int)WeaponType::Bomb || f.second == (int)WeaponType::ProximityBomb ? bombs : bullets)++;
    printf("  side %d fired in the simulator: %d bullets, %d bombs; first at %u\n", k, bullets, bombs,
           sim_fires[k].empty() ? 0 : sim_fires[k].front().first);
  }
  printf("  upgrade syncs: %d / %d; teleports re-synced: %d / %d\n", upgrade_syncs[0], upgrade_syncs[1], teleports[0], teleports[1]);
  printf("  deaths: real %d / %d, simulated %d / %d\n", real_deaths[0], real_deaths[1], sim_deaths[0], sim_deaths[1]);
  // Hits: each real hit matched to a simulated one on the same side, same weapon, within 20 ticks.
  std::vector<HitRecord> real = ReadRealHits(argv[4], 0), real_b = ReadRealHits(argv[7], 1);
  real.insert(real.end(), real_b.begin(), real_b.end());
  if (getenv("SIM_DEBUG")) {
    for (size_t k = 0; k < std::min<size_t>(6, real.size()); ++k) fprintf(stderr, "real hit %u victim %d weapon %d\n", real[k].tick, real[k].victim, real[k].weapon);
    for (size_t k = 0; k < std::min<size_t>(6, hits.size()); ++k) fprintf(stderr, "sim  hit %u victim %d weapon %d\n", hits[k].tick, hits[k].victim, hits[k].weapon);
  }
  std::vector<bool> used(hits.size(), false);
  int matched = 0;
  for (const HitRecord& r : real) {
    for (size_t k = 0; k < hits.size(); ++k) {
      if (used[k] || hits[k].victim != r.victim || hits[k].weapon != r.weapon) continue;
      if (std::abs((s32)(hits[k].tick - r.tick)) <= 20) {
        used[k] = true;
        ++matched;
        break;
      }
    }
  }
  if (getenv("SIM_DEBUG")) {
    for (const HitRecord& r : real) {
      bool found = false;
      for (size_t k = 0; k < hits.size(); ++k)
        if (hits[k].victim == r.victim && hits[k].weapon == r.weapon && std::abs((s32)(hits[k].tick - r.tick)) <= 20) found = true;
      if (!found) fprintf(stderr, "unmatched real hit %u victim %d class %d\n", r.tick, r.victim, r.weapon);
    }
  }
  if (getenv("SIM_DEBUG")) {
    for (size_t k = 0; k < hits.size(); ++k) if (!used[k]) fprintf(stderr, "unmatched sim hit %u victim %d class %d\n", hits[k].tick, hits[k].victim, hits[k].weapon);
  }
  printf("  hits: real %zu, simulated %zu, matched %d (%.0f%% of real, %.0f%% of simulated)\n", real.size(), hits.size(),
         matched, real.empty() ? 0.0 : 100.0 * matched / real.size(), hits.empty() ? 0.0 : 100.0 * matched / hits.size());
  return 0;
}

// --- A player's recording (bot-ai ticket 46): fair observations of what the player saw -----------------------------
// One client is fed the player's recorded packets at their own times (nullspace's NULLSPACE_CAPTURE), with its ship
// put where the player's was every tick (NULLSPACE_INPUTLOG): the player's own movement isn't simulated, so nullspace
// updating once a frame doesn't matter. The opponent and its weapons come from the server's packets, as they reached
// the player. Its fair observation is written every kObsEvery ticks, for pairing with the player's inputs.

struct SelfHitWatcher : EventHandler<WeaponSelfDamageEvent> {
  Game* game = nullptr;
  std::vector<HitRecord> hits, own;  // hits taken from others, and from the player's own weapons
  void HandleEvent(const WeaponSelfDamageEvent& event) override {
    if (!game) return;
    Player* self = game->player_manager.GetSelf();
    (self && &event.shooter == self ? own : hits).push_back({GetCurrentTick(), 0, WeaponClass(event.weapon.data)});
  }
};

// A player's real hits taken, from their combat log ("hit" events, in server ticks): from others, and their own.
void ReadPlayerHits(const char* path, const char* self_name, std::vector<HitRecord>& hits, std::vector<HitRecord>& own) {
  FILE* f = fopen(path, "r");
  if (!f) return;
  char line[4096], shooter[64];
  snprintf(shooter, sizeof(shooter), "\"shooter\":\"%s\"", self_name);
  while (fgets(line, sizeof(line), f)) {
    if (!strstr(line, "\"e\":\"hit\"")) continue;
    const char* t = strstr(line, "\"tick\":");
    const char* w = strstr(line, "\"weapon\":\"");
    if (!t || !w) continue;
    w += 10;
    int weapon = strncmp(w, "bullet", 6) == 0 || strncmp(w, "bounce", 6) == 0 ? 0
                 : strncmp(w, "bomb", 4) == 0 || strncmp(w, "prox", 4) == 0 || strncmp(w, "thor", 4) == 0 ? 1
                                                                                               : 2;
    (strstr(line, shooter) ? own : hits).push_back({(u32)strtoul(t + 7, nullptr, 10), 0, weapon});
  }
  fclose(f);
}

// Real hits matched to simulated ones: same weapon class, within 20 ticks.
int MatchHits(const std::vector<HitRecord>& real, const std::vector<HitRecord>& sim) {
  std::vector<bool> used(sim.size(), false);
  int matched = 0;
  for (const HitRecord& r : real) {
    for (size_t k = 0; k < sim.size(); ++k) {
      if (used[k] || sim[k].weapon != r.weapon || std::abs((s32)(sim[k].tick - r.tick)) > 20) continue;
      used[k] = true;
      ++matched;
      break;
    }
  }
  return matched;
}

// The enemy the observation is of: the first other player in a ship on another team.
u16 Opponent(Game& g) {
  Player* self = g.player_manager.GetSelf();
  if (!self) return 0xFFFF;
  for (size_t k = 0; k < g.player_manager.player_count; ++k) {
    Player& p = g.player_manager.players[k];
    if (&p != self && p.ship < 8 && p.frequency != self->frequency) return p.id;
  }
  return 0xFFFF;
}

// The player's ship as logged.
void PutShip(Player& self, const LoggedUpdate& u) {
  self.position = Vector2f(u.x, u.y);
  self.velocity = Vector2f(u.vx, u.vy);
  self.orientation = u.orientation;
  self.energy = u.energy;
  self.enter_delay = u.ship < 8 ? u.enter_delay : self.enter_delay;
}

int PlayerReplay(int argc, char** argv) {
  // player <capture> <input log> <out> [combat log]
  std::vector<Packet> capture = ReadCapture(argv[2]);
  s32 offset = ServerOffset(capture);
  std::vector<LoggedUpdate> log = ReadInputLog(argv[3], offset);
  if (capture.empty() || log.empty()) {
    fprintf(stderr, "can't read the recording\n");
    return 1;
  }
  size_t first = 0;
  while (first < log.size() && log[first].ship >= 8) ++first;
  if (first == log.size()) {
    fprintf(stderr, "the player never entered a ship\n");
    return 1;
  }
  kPlayerName = "sim";
  u32 start = log[first].tick, end = log.back().tick;
  SetVirtualTick(start);
  Relay relay;  // the client's own packets go nowhere
  static Side side;
  side.relay = &relay;
  relay.sides[0] = relay.sides[1] = &side;
  size_t next_packet = 0;
  if (!SetupSide(side, capture, start, offset, true, start, &next_packet)) {
    fprintf(stderr, "setup failed\n");
    return 1;
  }
  relay.outbox.clear();
  Game& g = *side.game;
  SelfHitWatcher watcher;
  watcher.game = &g;

  FILE* out = fopen(argv[4], "wb");
  if (!out) {
    fprintf(stderr, "can't write %s\n", argv[4]);
    return 1;
  }
  // The file: "ZRO1", the observation size u32, the server-time offset s32, then each observation as its local tick
  // (the recording's clock, as the input log's) u32 and its floats.
  u32 size = kFairSize;
  fwrite("ZRO1", 1, 4, out);
  fwrite(&size, 4, 1, out);
  fwrite(&offset, 4, 1, out);
  std::vector<float> obs(kFairSize);
  size_t next = first, delivered = 0, written = 0, on_screen = 0, on_radar = 0, alive = 0;
  const LoggedUpdate* now = &log[first];
  for (u32 t = start; TICK_GTE(end, t); ++t) {
    SetVirtualTick(t);
    while (side.next_deferred < side.deferred.size() && TICK_GTE(t, side.deferred[side.next_deferred].first)) {
      auto& pkt = side.deferred[side.next_deferred++].second;
      g.connection.ProcessPacket(pkt.data(), pkt.size());
    }
    // The server's packets due by now. Core packets (reliable, clusters) are skipped: their contents were captured
    // on their own as the client unwrapped them.
    while (next_packet < capture.size()) {
      const Packet& p = capture[next_packet];
      if (TICK_GT((u32)((s32)p.tick + offset) & 0x7fffffff, t)) break;
      ++next_packet;
      if (p.direction != 0 || p.data.empty() || p.data[0] == 0x00) continue;
      std::vector<u8> copy = p.data;
      g.connection.ProcessPacket(copy.data(), copy.size());
      ++delivered;
    }
    while (next < log.size() && TICK_GTE(t, log[next].tick)) now = &log[next++];
    Player* self = g.player_manager.GetSelf();
    if (!self) continue;
    if (now->ship < 8) {
      u32 fields[kShipFields];
      ShipFields(g.ship_controller.ship, fields);
      if (memcmp(fields, now->fields, sizeof(fields)) != 0) SetShipFields(g.ship_controller.ship, now->fields);
    }
    PutShip(*self, *now);
    side.input.actions = now->actions;
    g.Update(side.input, 0.01f);
    g.player_manager.kdtree = nullptr;
    side.trans.Revert(side.scratch);
    relay.outbox.clear();
    PutShip(*self, *now);
    if (t % kObsEvery == 0) {
      FairObserve(g, Opponent(g), t, obs.data());
      u32 local = (u32)((s32)t - offset) & 0x7fffffff;
      fwrite(&local, 4, 1, out);
      fwrite(obs.data(), 4, kFairSize, out);
      ++written;
      alive += obs[7] > 0.5f;
      on_screen += obs[8] > 0.5f;
      on_radar += obs[9] > 0.5f;
    }
  }
  fclose(out);
  printf("replayed %.1f min: %zu packets delivered, %zu observations written (alive %.0f%%, the opponent on screen "
         "%.0f%%, on radar only %.0f%%)\n", TICK_DIFF(end, start) / 6000.0f, delivered, written,
         100.0f * alive / std::max<size_t>(written, 1), 100.0f * on_screen / std::max<size_t>(written, 1),
         100.0f * on_radar / std::max<size_t>(written, 1));
  if (argc > 5) {
    // A check that the opponent's weapons arrived as the player saw them: the hits the client took against the real
    // ones from the player's combat log.
    std::vector<HitRecord> real, real_own;
    Player* self = g.player_manager.GetSelf();
    ReadPlayerHits(argv[5], self ? self->name : "", real, real_own);
    printf("  hits taken from the opponent: real %zu, replayed %zu, matched %d; from own weapons: real %zu, replayed %zu "
           "(own weapons are the player's, fired in the replay from their keys)\n",
           real.size(), watcher.hits.size(), MatchHits(real, watcher.hits), real_own.size(), watcher.own.size());
  }
  return 0;
}
}  // namespace


// --- A C API for training (ticket 41): environments of two clients each, stepped by callers -----------------------
// Each environment keeps its own clock; the process-wide virtual clock is set to it before any of its work, so
// environments in one process need not step together. Built as a shared library (DUELSIM_LIBRARY) for Python ctypes.

namespace {

struct Env {
  Relay relay;
  Side sides[2];
  u32 tick = 0;
  // zero draws hit outcomes (a prize lost to damage) from the C library's rand(), one sequence for the process. Each
  // tick reseeds it from this and the tick, so an environment plays the same whatever else shares its process.
  u32 seed = 0;
  // Fair observations kept every kObsEvery ticks, per side: (tick, observation), the latest last.
  std::deque<std::pair<u32, std::vector<float>>> fair[2];
  struct Happening {
    int kind;  // 0 a hit taken, 1 a death, 2 a hit taken from one's own weapon
    int side;  // the side hit, or that died
    float value;  // damage, or the killer's side
    int weapon;   // the weapon's class, for hits
  };
  std::vector<Happening> happenings;
};

std::vector<Env*> envs;

struct EnvWatcher : EventHandler<WeaponSelfDamageEvent>, EventHandler<PlayerDeathEvent> {
  void HandleEvent(const WeaponSelfDamageEvent& event) override {
    for (Env* env : envs) {
      for (int i = 0; i < 2; ++i) {
        PlayerManager& pm = env->sides[i].game->player_manager;
        if (&event.shooter >= pm.players && &event.shooter < pm.players + pm.player_count) {
          int kind = event.shooter.id == env->sides[i].pid ? 2 : 0;  // 2: its own weapon
          env->happenings.push_back({kind, i, (float)event.damage, WeaponClass(event.weapon.data)});
          return;
        }
      }
    }
  }
  void HandleEvent(const PlayerDeathEvent& event) override {
    // Each client hears every kill; side 0's client reports them.
    for (Env* env : envs) {
      PlayerManager& pm = env->sides[0].game->player_manager;
      if (&event.player < pm.players || &event.player >= pm.players + pm.player_count) continue;
      int dead = event.player.id == env->sides[0].pid ? 0 : 1;
      int killer = event.killer.id == env->sides[0].pid ? 0 : 1;
      env->happenings.push_back({1, dead, (float)killer, 0});
      return;
    }
  }
};

EnvWatcher* env_watcher = nullptr;

void Deliver(Env& env) {
  while (!env.relay.outbox.empty()) {
    auto [to, pkt] = env.relay.outbox.front();
    env.relay.outbox.pop_front();
    InScope(env.sides[to], [&] { env.sides[to].game->connection.ProcessPacket(pkt.data(), pkt.size()); });
  }
}

}  // namespace

extern "C" {

void* ds_create_bot(const char* capture_a, const char* capture_b, const char* bot_config, const char* bot_args,
                    unsigned int seed);

// A new environment from two captured logins (one per side), run from a folder holding zones/<server>/<map>.lvl.
// Null on failure.
void* ds_create(const char* capture_a, const char* capture_b) { return ds_create_bot(capture_a, capture_b, nullptr, nullptr, 0); }

// As ds_create, with side 1 flown by a zero Chaos bot (ticket 45): its bot config file (zero's chaos.cfg) and
// arguments as on zero's command line ("--skill veteran --aggression standard --ship 1"); its random generators are
// seeded from `seed`. The bot side's actions in ds_step are ignored. Without a config, as ds_create.
void* ds_create_bot(const char* capture_a, const char* capture_b, const char* bot_config, const char* bot_args,
                    unsigned int seed) {
  static unsigned int next_scope = 1;
  auto a = ReadCapture(capture_a), b = ReadCapture(capture_b);
  if (a.empty() || b.empty()) return nullptr;
  if (!env_watcher) env_watcher = new EnvWatcher();  // scope 0: it hears every environment
  g_LogPrintLevel = LogLevel::Error;  // the clients' packet chatter would swamp a training run
  Env* env = new Env();
  env->tick = 100000;
  SetVirtualTick(env->tick);
  for (int i = 0; i < 2; ++i) {
    env->sides[i].index = i;
    env->sides[i].scope = next_scope++;
    env->sides[i].relay = &env->relay;
    env->relay.sides[i] = &env->sides[i];
  }
  if (bot_config && *bot_config) {
    env->sides[1].bot_config = bot_config;
    env->sides[1].bot_args = bot_args ? bot_args : "";
    env->sides[1].bot_seed = seed * 7919u + 1;
  }
  if (!SetupSide(env->sides[0], a, env->tick) || !SetupSide(env->sides[1], b, env->tick)) return nullptr;
  env->relay.outbox.clear();
  envs.push_back(env);
  return env;
}

// Puts both ships at the given positions (tiles), still, at full energy, facing `orientation` (0-1).
void ds_reset(void* handle, float ax, float ay, float a_orientation, float bx, float by, float b_orientation) {
  Env& env = *(Env*)handle;
  SetVirtualTick(env.tick);
  const float pos[2][3] = {{ax, ay, a_orientation}, {bx, by, b_orientation}};
  for (int i = 0; i < 2; ++i) {
    Game& g = *env.sides[i].game;
    Player* self = g.player_manager.GetSelf();
    if (!self) continue;
    self->position = Vector2f(pos[i][0], pos[i][1]);
    self->velocity = Vector2f(0, 0);
    self->orientation = pos[i][2];
    self->enter_delay = 0.0f;
    self->energy = (float)g.ship_controller.ship.energy;
  }
  // The last episode's weapons still in flight, in both clients' views.
  for (int i = 0; i < 2; ++i) {
    PlayerManager& pm = env.sides[i].game->player_manager;
    InScope(env.sides[i], [&] {
      for (size_t k = 0; k < pm.player_count; ++k) env.sides[i].game->weapon_manager.ClearWeapons(pm.players[k]);
    });
  }
  env.happenings.clear();
  env.fair[0].clear();
  env.fair[1].clear();
}

// Steps `ticks` ticks (10 ms each) with each side's held actions (InputAction bits).
void ds_step(void* handle, unsigned int actions_a, unsigned int actions_b, int ticks) {
  Env& env = *(Env*)handle;
  for (int t = 0; t < ticks; ++t) {
    SetVirtualTick(++env.tick);
    srand(env.seed * 2654435761u + env.tick);
    const u32 actions[2] = {actions_a, actions_b};
    for (int i = 0; i < 2; ++i) {
      Side& side = env.sides[i];
      InScope(side, [&] {
        if (side.bot) {
          BotThink(side);  // a bot side flies itself: its tree's input, not the caller's
        } else {
          side.input.actions = actions[i];
        }
        side.game->Update(side.input, 0.01f);
        side.game->player_manager.kdtree = nullptr;
        side.trans.Revert(side.scratch);
      });
    }
    Deliver(env);
    if (env.tick % kObsEvery == 0) {
      for (int i = 0; i < 2; ++i) {
        std::vector<float> o(kFairSize);
        FairObserve(*env.sides[i].game, env.sides[1 - i].pid, env.tick, o.data());
        env.fair[i].push_back({env.tick, std::move(o)});
        while (env.fair[i].size() > 16) env.fair[i].pop_front();
      }
    }
  }
}

void ds_seed(void* handle, unsigned int seed) { ((Env*)handle)->seed = seed; }

// Where side 1's bot flies, with --behavior bench (the piloting test bench, ticket 45): the "bench_goal" position.
void ds_bot_goal(void* handle, float x, float y) {
  Side& side = ((Env*)handle)->sides[1];
  if (side.bot) side.bot->execute_ctx.blackboard.Set("bench_goal", Vector2f(x, y));
}

// A side's ship upgrades, as zero's input log has them (InputLog.h: kShipFields u32s, energy to capability).
void ds_ship_fields(void* handle, int side_index, unsigned int* out) {
  ShipFields(((Env*)handle)->sides[side_index].game->ship_controller.ship, out);
}

int ds_fair_size() { return kFairSize; }

// zero's own run of a policy file (zero/policy/DuelPolicy.h), for checking it against PyTorch: null if it won't load.
const void* ds_policy_load(const char* path) { return LoadSharedPolicy(path); }

// One step of a loaded policy: obs (its obs size), state (its hidden size, updated) and logits (DuelPolicy::kLogits).
void ds_policy_step(const void* policy, const float* obs, float* state, float* logits) {
  ((const DuelPolicy*)policy)->Step(obs, state, logits);
}

// A side's fair observation as it was kObsDelayTicks ago (the reaction delay): kFairSize floats (ds_fair_size).
// Returns 0 until there's one that old.
int ds_fair_observe(void* handle, int side_index, float* out) {
  Env& env = *(Env*)handle;
  const auto& history = env.fair[side_index];
  for (auto it = history.rbegin(); it != history.rend(); ++it) {
    if (TICK_DIFF(env.tick, it->first) >= (s32)kObsDelayTicks) {
      std::copy(it->second.begin(), it->second.end(), out);
      return 1;
    }
  }
  return 0;
}

// One side's view, as 18 + 5 x max_weapons floats: own x, y, vx, vy, orientation, energy, max energy, enter delay
// (s), ticks until its bullet and bomb are ready; the opponent as this client sees it: x, y, vx, vy, orientation,
// enter delay, energy as last reported, and 1 if it's in the game; then up to max_weapons enemy weapons nearest the
// ship: x, y, vx, vy, class (0 bullet, 1 bomb, 2 other), unused slots -1. Returns how many weapons were written.
int ds_observe(void* handle, int side_index, float* out, int max_weapons) {
  Env& env = *(Env*)handle;
  Side& side = env.sides[side_index];
  Game& g = *side.game;
  Player* self = g.player_manager.GetSelf();
  Player* them = g.player_manager.GetPlayerById(env.sides[1 - side_index].pid);
  int n = 0;
  auto put = [&](float v) { out[n++] = v; };
  Ship& ship = g.ship_controller.ship;
  put(self ? self->position.x : 0), put(self ? self->position.y : 0), put(self ? self->velocity.x : 0);
  put(self ? self->velocity.y : 0), put(self ? self->orientation : 0), put(self ? self->energy : 0);
  put((float)ship.energy), put(self ? self->enter_delay : 0);
  put((float)std::max<s32>(0, TICK_DIFF(ship.next_bullet_tick, env.tick)));
  put((float)std::max<s32>(0, TICK_DIFF(ship.next_bomb_tick, env.tick)));
  put(them ? them->position.x : 0), put(them ? them->position.y : 0), put(them ? them->velocity.x : 0);
  put(them ? them->velocity.y : 0), put(them ? them->orientation : 0), put(them ? them->enter_delay : 0);
  put(them ? them->energy : 0), put(them && them->ship < 8 ? 1.0f : 0.0f);
  std::vector<std::pair<float, const Weapon*>> near;
  WeaponManager& wm = g.weapon_manager;
  for (size_t k = 0; k < wm.weapon_count; ++k) {
    const Weapon& w = wm.weapons[k];
    if (self && w.frequency != self->frequency) near.push_back({w.position.DistanceSq(self->position), &w});
  }
  std::sort(near.begin(), near.end(), [](auto& x, auto& y) { return x.first < y.first; });
  int written = 0;
  for (int k = 0; k < max_weapons; ++k) {
    if (k < (int)near.size()) {
      const Weapon& w = *near[k].second;
      put(w.position.x), put(w.position.y), put(w.velocity.x), put(w.velocity.y), put((float)WeaponClass(w.data));
      ++written;
    } else {
      for (int j = 0; j < 5; ++j) put(-1.0f);
    }
  }
  return written;
}

// What happened since the last call: up to `max` records of 4 floats (kind 0 hit / 1 death / 2 hit by one's own
// weapon, side, damage or killer side, weapon class). Returns how many.
int ds_happenings(void* handle, float* out, int max) {
  Env& env = *(Env*)handle;
  int n = 0;
  for (auto& h : env.happenings) {
    if (n >= max) break;
    out[n * 4] = (float)h.kind, out[n * 4 + 1] = (float)h.side, out[n * 4 + 2] = h.value, out[n * 4 + 3] = (float)h.weapon;
    ++n;
  }
  env.happenings.clear();
  return n;
}

// Whether a tile is a wall for a side's ship (doors as they are now).
int ds_solid(void* handle, int side_index, int x, int y) {
  Env& env = *(Env*)handle;
  Game& g = *env.sides[side_index].game;
  Player* self = g.player_manager.GetSelf();
  return g.GetMap().IsSolid((u16)x, (u16)y, self ? self->frequency : 0) ? 1 : 0;
}

}  // extern "C"

#ifndef DUELSIM_LIBRARY
int main(int argc, char** argv) {
  if (argc >= 8 && strcmp(argv[1], "replay") == 0) return Replay(argv);
  if (argc >= 5 && strcmp(argv[1], "player") == 0) return PlayerReplay(argc, argv);
  if (argc < 4) {
    fprintf(stderr, "usage: duelsim <capture A> <capture B> <seconds>\n");
    return 1;
  }
  auto a = ReadCapture(argv[1]), b = ReadCapture(argv[2]);
  float seconds = (float)atof(argv[3]);
  if (a.empty() || b.empty()) {
    fprintf(stderr, "can't read the captures\n");
    return 1;
  }
  kPlayerName = "sim";
  Tick start = 100000;
  SetVirtualTick(start);

  Relay relay;
  Side sides[2];
  for (int i = 0; i < 2; ++i) {
    sides[i].index = i;
    sides[i].relay = &relay;
    relay.sides[i] = &sides[i];
  }
  if (!SetupSide(sides[0], a, start) || !SetupSide(sides[1], b, start)) {
    fprintf(stderr, "setup failed (player ids %d, %d)\n", sides[0].pid, sides[1].pid);
    return 1;
  }
  relay.outbox.clear();
  printf("setup: side A is player %d, side B player %d\n", sides[0].pid, sides[1].pid);

  // A scripted fight for this checkpoint: both thrust and turn, A fires bullets every so often.
  u32 ticks = (u32)(seconds * 100.0f);
  auto wall = std::chrono::steady_clock::now();
  for (u32 t = 1; t <= ticks; ++t) {
    SetVirtualTick(start + t);
    for (int i = 0; i < 2; ++i) {
      InputState& in = sides[i].input;
      in.Clear();
      in.SetAction(InputAction::Forward, (t / 150) % 2 == 0);
      in.SetAction(i == 0 ? InputAction::Left : InputAction::Right, (t / 70) % 3 == 0);
      in.SetAction(InputAction::Bullet, i == 0 && (t % 40) < 5);
      sides[i].game->Update(in, 0.01f);
      // As zero's own loop does after every update: the collision tree is rebuilt, the scratch memory freed.
      sides[i].game->player_manager.kdtree = nullptr;
      sides[i].trans.Revert(sides[i].scratch);
    }
    while (!relay.outbox.empty()) {
      auto [to, pkt] = relay.outbox.front();
      relay.outbox.pop_front();
      sides[to].game->connection.ProcessPacket(pkt.data(), pkt.size());
    }
    if (t % 200 == 0) {
      for (int i = 0; i < 2; ++i) {
        Game& g = *sides[i].game;
        Player* self = g.player_manager.GetSelf();
        Player* other = g.player_manager.GetPlayerById(sides[1 - i].pid);
        if (!self) continue;
        printf("t=%5.1fs side %c: self (%.1f, %.1f) energy %.0f ship %d | sees other at (%.1f, %.1f)\n", t / 100.0f,
               'A' + i, self->position.x, self->position.y, self->energy, self->ship,
               other ? other->position.x : -1.0f, other ? other->position.y : -1.0f);
      }
    }
  }
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count();
  printf("simulated %.0f s in %.3f s wall: %.0f times real time\n", seconds, elapsed, seconds / elapsed);
  return 0;
}
#endif  // DUELSIM_LIBRARY
