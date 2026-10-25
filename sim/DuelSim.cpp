// A headless duel simulator (bot-ai ticket 41), first checkpoint: two zero game clients in one process, on a virtual
// clock, with an in-memory relay in place of the server. Each client simulates its own ship and takes its own damage,
// as in the real game; the relay passes positions (with any weapon fired) between them and turns deaths into kills.
// Each client's arena setup is replayed from packets captured from a real game (zero's ZERO_CAPTURE).
//
// duelsim <capture A> <capture B> <seconds>   (run from a bots folder holding zones/Chaos/<map>.lvl)
#include <zero/game/Clock.h>
#include <zero/game/Game.h>
#include <zero/game/InputState.h>
#include <zero/game/Logger.h>
#include <zero/game/Memory.h>
#include <zero/game/WorkQueue.h>
#include <zero/game/net/Connection.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

using namespace zero;

namespace zero {
// Globals the client expects: the bot's name (ZeroBot.cpp), and the server name, whose folder (zones/<name>) holds the
// cached map, and the security service address (normally defined with the bot's main).
extern const char* kPlayerName;
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
  Relay* relay = nullptr;
  u16 pid = 0xFFFF;
  MemoryArena perm{nullptr, 0}, trans{nullptr, 0}, work{nullptr, 0};
  WorkQueue* queue = nullptr;
  Worker* worker = nullptr;
  Game* game = nullptr;
  InputState input;
};

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

bool SetupSide(Side& side, const std::vector<Packet>& capture, Tick start) {
  constexpr size_t kPermanentSize = Megabytes(64), kTransientSize = Megabytes(32), kWorkSize = Megabytes(4);
  side.perm = MemoryArena((u8*)malloc(kPermanentSize), kPermanentSize);
  side.trans = MemoryArena((u8*)malloc(kTransientSize), kTransientSize);
  side.work = MemoryArena((u8*)malloc(kWorkSize), kWorkSize);
  side.queue = new WorkQueue(side.work);
  side.worker = new Worker(*side.queue);
  side.worker->Launch();
  side.game = memory_arena_construct_type(&side.perm, Game, side.perm, side.trans, *side.queue, 1920, 1080);
  if (side.game->Initialize(side.input) == GameInitializeResult::Failure) return false;
  Connection& c = side.game->connection;
  c.loopback = Loopback;
  c.loopback_user = &side;
  c.connected = true;
  // Replay the arena setup: every server packet but positions, kills, chat and keep-alives.
  for (const Packet& p : capture) {
    if (p.direction != 0 || p.data.empty()) continue;
    u8 type = p.data[0];
    if (type == 0x00 || type == 0x05 || type == 0x28 || type == 0x06 || type == 0x07 || type == 0x27) continue;
    if (type == 0x01 && p.data.size() >= 3) side.pid = Read16(p.data.data() + 1);
    SetVirtualTick(start);
    std::vector<u8> copy = p.data;
    c.ProcessPacket(copy.data(), copy.size());
  }
  c.time_diff = 0;
  return side.pid != 0xFFFF;
}

}  // namespace

int main(int argc, char** argv) {
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
