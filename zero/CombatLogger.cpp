#include "CombatLogger.h"

#include <zero/HeuristicEnergyTracker.h>
#include <zero/ZeroBot.h>
#include <zero/game/Game.h>
#include <zero/game/Logger.h>

namespace zero {

// The bot writes its position, velocity and energy this often (server ticks: half a second).
constexpr u32 kEstimateInterval = 50;
// An enemy bomb exploding within this many blast radii of the bot, without hurting it, counts as dodged.
constexpr float kNearMissRadii = 1.5f;

static const char* GetWeaponName(WeaponData data) {
  switch (data.type) {
    case WeaponType::Bullet:
    case WeaponType::BouncingBullet:
      // Shrapnel is a bullet that carries its bomb's shrapnel count.
      return data.shrap > 0 ? "shrapnel" : "bullet";
    case WeaponType::Bomb:
    case WeaponType::ProximityBomb:
      return data.alternate ? "mine" : "bomb";
    case WeaponType::Repel:
      return "repel";
    case WeaponType::Decoy:
      return "decoy";
    case WeaponType::Burst:
      return "burst";
    case WeaponType::Thor:
      return "thor";
    default:
      return "other";
  }
}

static bool IsBomb(WeaponData data) {
  return data.type == WeaponType::Bomb || data.type == WeaponType::ProximityBomb || data.type == WeaponType::Thor;
}

// Writes a player name as a JSON string, escaping quotes, backslashes and control characters.
static void WriteJsonString(FILE* file, const char* text) {
  fputc('"', file);
  for (const char* c = text; *c; ++c) {
    if (*c == '"' || *c == '\\') {
      fprintf(file, "\\%c", *c);
    } else if ((unsigned char)*c < 0x20) {
      fprintf(file, "\\u%04x", (unsigned char)*c);
    } else {
      fputc(*c, file);
    }
  }
  fputc('"', file);
}

static float GetWeaponSpeedTiles(Game& game, Player& player, WeaponData data) {
  const ShipSettings& settings = game.connection.settings.ShipSettings[player.ship];
  switch (data.type) {
    case WeaponType::Bullet:
    case WeaponType::BouncingBullet:
      return settings.BulletSpeed / 16.0f / 10.0f;
    case WeaponType::Bomb:
    case WeaponType::ProximityBomb:
    case WeaponType::Thor:
      return data.alternate ? 0.0f : settings.BombSpeed / 16.0f / 10.0f;
    default:
      return 0.0f;
  }
}

// What firing actually cost the bot, as ShipController charges it: free with super, scaled while carrying a flag.
static u32 GetSelfFireCost(Game& game, Player& self, WeaponData data) {
  if (game.ship_controller.ship.super_time > 0.0f) return 0;
  u32 cost = GetWeaponFireCost(game.connection.settings.ShipSettings[self.ship], data);
  bool bullet_or_bomb = data.type == WeaponType::Bullet || data.type == WeaponType::BouncingBullet ||
                        data.type == WeaponType::Bomb || data.type == WeaponType::ProximityBomb;
  if (bullet_or_bomb && (self.flags > 0 || (self.ball_carrier && game.connection.settings.UseFlagger))) {
    cost = (u32)(cost * (game.connection.settings.FlaggerFireCostPercent / 1000.0f));
  }
  return cost;
}

// An enemy bomb near the bot, and what became of it: "hit", "repelled" or "dodged".
static void WriteBombEvent(FILE* file, u32 tick, const char* shooter, const char* outcome, float closest) {
  fprintf(file, "{\"e\":\"bomb\",\"tick\":%u,\"shooter\":", tick);
  WriteJsonString(file, shooter);
  fprintf(file, ",\"outcome\":\"%s\",\"closest\":%.2f}\n", outcome, closest);
}

static float GetBlastRadiusTiles(Game& game, WeaponData data) {
  float pixels = (float)game.connection.settings.BombExplodePixels * (data.level + 1);
  return pixels / 16.0f;
}

CombatLogger::CombatLogger(ZeroBot& bot, const char* path, u32 state_ticks) : bot(bot), state_ticks(state_ticks) {
  file = fopen(path, "a");
  if (!file) {
    Log(LogLevel::Error, "Failed to open combat event log %s", path);
    return;
  }
  // The first spawn line is written when the bot enters a ship (the ship change event).
}

CombatLogger::~CombatLogger() {
  if (file) fclose(file);
}

u32 CombatLogger::ServerTick() {
  return bot.game ? bot.game->connection.GetServerTick() : 0;
}

// Ship is 1-8, or 9 for spectator.
void CombatLogger::WriteSpawn(int ship, int freq) {
  if (!file) return;
  fprintf(file, "{\"e\":\"spawn\",\"tick\":%u,\"name\":", ServerTick());
  WriteJsonString(file, bot.name);
  fprintf(file, ",\"ship\":%d,\"freq\":%d}\n", ship, freq);
  fflush(file);
}

void CombatLogger::Update() {
  if (!file || !bot.game) return;
  Player* self = bot.game->player_manager.GetSelf();
  if (!self || self->ship >= 8) return;

  u32 tick = ServerTick();
  if (TICK_GT(next_state_tick, tick)) return;
  next_state_tick = tick + state_ticks;

  fprintf(file,
          "{\"e\":\"state\",\"tick\":%u,\"x\":%.2f,\"y\":%.2f,\"vx\":%.2f,\"vy\":%.2f,\"energy\":%d,"
          "\"max_energy\":%u}\n",
          tick, self->position.x, self->position.y, self->velocity.x, self->velocity.y, (int)self->energy,
          bot.game->ship_controller.ship.energy);
  fflush(file);
}

void CombatLogger::WriteEstimate(const char* target, float energy, float sd, float heuristic) {
  if (!file) return;

  u32 tick = ServerTick();
  if (TICK_GT(next_estimate_tick, tick)) return;
  next_estimate_tick = tick + kEstimateInterval;

  fprintf(file, "{\"e\":\"estimate\",\"tick\":%u,\"target\":", tick);
  WriteJsonString(file, target);
  fprintf(file, ",\"energy\":%d,\"sd\":%d,\"heuristic\":%d}\n", (int)energy, (int)sd, (int)heuristic);
  fflush(file);
}

void CombatLogger::WriteTarget(const char* target, const char* reason) {
  if (!file) return;

  fprintf(file, "{\"e\":\"target\",\"tick\":%u,\"target\":", ServerTick());
  if (target) {
    WriteJsonString(file, target);
  } else {
    fprintf(file, "null");
  }
  fprintf(file, ",\"reason\":");
  WriteJsonString(file, reason);
  fprintf(file, "}\n");
  fflush(file);
}

void CombatLogger::WriteMode(const char* mode, float win_chance) {
  if (!file) return;

  fprintf(file, "{\"e\":\"mode\",\"tick\":%u,\"mode\":", ServerTick());
  WriteJsonString(file, mode);
  fprintf(file, ",\"win_chance\":%.2f}\n", win_chance);
  fflush(file);
}

void CombatLogger::WriteAccept(float damage, float win_chance) {
  if (!file) return;

  fprintf(file, "{\"e\":\"accept\",\"tick\":%u,\"damage\":%d,\"win_chance\":%.2f}\n", ServerTick(), (int)damage,
          win_chance);
  fflush(file);
}

void CombatLogger::WriteTactic(const char* name, const char* target) {
  if (!file) return;

  fprintf(file, "{\"e\":\"tactic\",\"tick\":%u,\"name\":", ServerTick());
  WriteJsonString(file, name);
  fprintf(file, ",\"target\":");
  WriteJsonString(file, target);
  fprintf(file, "}\n");
  fflush(file);
}

void CombatLogger::HandleEvent(const WeaponFireEvent& event) {
  if (!file || !bot.game) return;
  Game& game = *bot.game;
  Player* self = game.player_manager.GetSelf();
  if (!self || &event.player != self || self->ship >= 8) return;

  u32 tick = ServerTick();
  u32 cost = GetSelfFireCost(game, *self, event.data);
  Vector2f shot_velocity = self->velocity + self->GetHeading() * GetWeaponSpeedTiles(game, *self, event.data);

  // The nearest enemy is what the shot is most likely aimed at.
  Player* target = nullptr;
  float target_distance = 0.0f;
  for (size_t i = 0; i < game.player_manager.player_count; ++i) {
    Player& other = game.player_manager.players[i];
    if (other.id == self->id || other.ship >= 8 || other.frequency == self->frequency) continue;
    float distance = other.position.Distance(self->position);
    if (!target || distance < target_distance) {
      target = &other;
      target_distance = distance;
    }
  }

  bool multifire =
      event.data.alternate && (event.data.type == WeaponType::Bullet || event.data.type == WeaponType::BouncingBullet);
  fprintf(file, "{\"e\":\"fire\",\"tick\":%u,\"weapon\":\"%s\",\"level\":%d,\"multifire\":%s,\"cost\":%u,", tick,
          GetWeaponName(event.data), event.data.level + 1, multifire ? "true" : "false", cost);
  fprintf(file, "\"energy\":%d,\"shot_speed\":%.2f,\"target\":", (int)self->energy, shot_velocity.Length());
  if (target) {
    WriteJsonString(file, target->name);
    fprintf(file, ",\"distance\":%.2f}\n", target_distance);
  } else {
    fprintf(file, "null,\"distance\":null}\n");
  }

  // A repel pushes away the enemy bombs within its reach: record each one as repelled.
  if (event.data.type == WeaponType::Repel) {
    float reach = game.connection.settings.RepelDistance / 16.0f;
    WeaponManager& weapons = game.weapon_manager;
    for (size_t i = 0; i < weapons.weapon_count; ++i) {
      Weapon& weapon = weapons.weapons[i];
      if (!IsBomb(weapon.data) || weapon.frequency == self->frequency) continue;
      float distance = weapon.GetPosition().Distance(self->position);
      if (distance > reach) continue;
      Player* shooter = game.player_manager.GetPlayerById(weapon.player_id);
      WriteBombEvent(file, tick, shooter ? shooter->name : "?", "repelled", distance);
    }
  }
  fflush(file);
}

void CombatLogger::HandleEvent(const WeaponSelfDamageEvent& event) {
  if (!file || !bot.game) return;
  Game& game = *bot.game;
  Player* self = game.player_manager.GetSelf();
  if (!self) return;

  u32 tick = ServerTick();
  bool bomb = IsBomb(event.weapon.data) && !event.weapon.data.alternate;
  float blast = event.weapon.GetPosition().Distance(self->position);

  fprintf(file, "{\"e\":\"hit\",\"tick\":%u,\"shooter\":", tick);
  WriteJsonString(file, event.shooter.name);
  fprintf(file, ",\"weapon\":\"%s\",\"level\":%d,\"damage\":%d,\"energy\":%d,\"distance\":%.2f,\"blast\":",
          GetWeaponName(event.weapon.data), event.weapon.data.level + 1, event.damage, (int)event.energy_before,
          event.shooter.position.Distance(self->position));
  if (bomb) {
    fprintf(file, "%.2f}\n", blast);
  } else {
    fprintf(file, "null}\n");
  }

  if (bomb && &event.shooter != self) {
    last_damaging_shooter = event.shooter.id;
    last_damage_tick = GetCurrentTick();
    WriteBombEvent(file, tick, event.shooter.name, "hit", blast);
  }
  fflush(file);
}

void CombatLogger::HandleEvent(const WeaponHitEvent& event) {
  if (!file || !bot.game) return;
  Game& game = *bot.game;
  Player* self = game.player_manager.GetSelf();
  if (!self || self->ship >= 8 || self->enter_delay > 0.0f) return;

  // An enemy bomb going off near the bot without hurting it: a near miss.
  Weapon& weapon = event.weapon;
  if (!IsBomb(weapon.data) || weapon.data.alternate || weapon.frequency == self->frequency) return;
  if (event.target == self) return;
  if (weapon.player_id == last_damaging_shooter && TICK_DIFF(GetCurrentTick(), last_damage_tick) <= 1) return;

  float distance = weapon.GetPosition().Distance(self->position);
  if (distance > GetBlastRadiusTiles(game, weapon.data) * kNearMissRadii) return;

  Player* shooter = game.player_manager.GetPlayerById(weapon.player_id);
  WriteBombEvent(file, ServerTick(), shooter ? shooter->name : "?", "dodged", distance);
  fflush(file);
}

void CombatLogger::HandleEvent(const PlayerDeathEvent& event) {
  if (!file || !bot.game) return;
  Player* self = bot.game->player_manager.GetSelf();
  if (!self) return;

  u32 tick = ServerTick();
  if (&event.player == self) {
    fprintf(file, "{\"e\":\"death\",\"tick\":%u,\"killer\":", tick);
    WriteJsonString(file, event.killer.name);
    fprintf(file, "}\n");
  } else if (&event.killer == self) {
    fprintf(file, "{\"e\":\"kill\",\"tick\":%u,\"victim\":", tick);
    WriteJsonString(file, event.player.name);
    fprintf(file, "}\n");
  }
  last_damaging_shooter = 0xFFFF;
  fflush(file);
}

void CombatLogger::HandleEvent(const PlayerFreqAndShipChangeEvent& event) {
  if (!bot.game || &event.player != bot.game->player_manager.GetSelf()) return;
  WriteSpawn(event.new_ship + 1, event.new_frequency);
}

}  // namespace zero
