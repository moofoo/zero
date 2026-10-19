#include "Opportunities.h"

#include <math.h>

#include <zero/CombatLogger.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

// How much likelier a shot is to hit a slow target, measured from Chaos games on 2026-10-19 (hits over shots, slow
// targets against all others).
constexpr float kSlowBulletFactor = 1.45f;
constexpr float kSlowBombNearFactor = 1.2f;
constexpr float kSlowBombFarFactor = 1.1f;
constexpr float kSlowBombNearDistance = 20.0f;
// Battle blasting: what counts as two other enemies fighting, and when it's over.
constexpr float kBattleSpread = 20.0f;
constexpr float kBattleNear = 30.0f;
constexpr float kBattleFar = 70.0f;
constexpr u32 kBattleTicks = 200;
constexpr u32 kBattleQuietTicks = 300;

Opportunities::Opportunities(ZeroBot& bot, float spotting, float notice, Vector2f view_extent)
    : bot(bot), spotting(spotting), notice(notice), view_extent(view_extent), random(std::random_device{}()) {}

bool Opportunities::OnScreen(const Player& player) const {
  Player* self = bot.game->player_manager.GetSelf();
  if (!self) return false;
  Vector2f delta = player.position - self->position;
  return fabsf(delta.x) <= view_extent.x && fabsf(delta.y) <= view_extent.y;
}

bool Opportunities::Roll(float chance) {
  if (chance >= 1.0f) return true;
  if (chance <= 0.0f) return false;
  return std::uniform_real_distribution<float>(0.0f, 1.0f)(random) < chance;
}

float Opportunities::WindowSeconds(const Player& player) {
  auto it = window_end.find(player.id);
  if (it == window_end.end()) return 0.0f;
  s32 left = TICK_DIFF(it->second, GetCurrentTick());
  return left > 0 ? left / 100.0f : 0.0f;
}

bool Opportunities::SpottedSlow(const Player& player) {
  Slowdown& slowdown = slowdowns[player.id];
  bool slow = player.velocity.Length() < kSlowSpeed && OnScreen(player);
  if (slow && !slowdown.slow) {
    slowdown.spotted = Roll(spotting);
    if (slowdown.spotted && bot.combat_logger) bot.combat_logger->WriteTactic("slow", player.name);
  }
  slowdown.slow = slow;
  return slow && slowdown.spotted;
}

float Opportunities::SlowFactor(const Player& player, bool bomb, float distance) {
  if (!SpottedSlow(player)) return 1.0f;
  if (!bomb) return kSlowBulletFactor;
  return distance < kSlowBombNearDistance ? kSlowBombNearFactor : kSlowBombFarFactor;
}

float Opportunities::SecondsSinceHit() const {
  if (!hit_yet) return 1e6f;
  return TICK_DIFF(GetCurrentTick(), last_hit_tick) / 100.0f;
}

float Opportunities::SecondsSinceFight() const {
  float dealt = dealt_yet ? TICK_DIFF(GetCurrentTick(), last_dealt_tick) / 100.0f : 1e6f;
  float hit = SecondsSinceHit();
  return dealt < hit ? dealt : hit;
}

std::optional<u32> Opportunities::LastHit(u16 shooter, u16 target) const {
  auto it = enemy_hits.find(((u32)shooter << 16) | target);
  if (it == enemy_hits.end()) return {};
  return it->second;
}

bool Opportunities::BattleGoing(u16 a, u16 b) const {
  u32 now = GetCurrentTick();
  auto recent = [&](std::optional<u32> tick) { return tick && TICK_DIFF(now, *tick) <= (s32)kBattleQuietTicks; };
  return recent(LastHit(a, b)) || recent(LastHit(b, a));
}

std::optional<std::pair<u16, u16>> Opportunities::FindBattle(const std::vector<Player*>& candidates,
                                                             const Player& self) {
  u32 now = GetCurrentTick();
  auto recent = [&](std::optional<u32> tick) { return tick && TICK_DIFF(now, *tick) <= (s32)kBattleTicks; };
  for (size_t i = 0; i < candidates.size(); ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      Player& a = *candidates[i];
      Player& b = *candidates[j];
      if (!recent(LastHit(a.id, b.id)) || !recent(LastHit(b.id, a.id))) continue;
      if (a.position.Distance(b.position) > kBattleSpread) continue;
      float da = a.position.Distance(self.position);
      float db = b.position.Distance(self.position);
      float nearer = da < db ? da : db;
      if (nearer < kBattleNear || nearer > kBattleFar) continue;
      return std::make_pair(a.id, b.id);
    }
  }
  return {};
}

void Opportunities::HandleEvent(const WeaponHitEvent& event) {
  Player* self = bot.game ? bot.game->player_manager.GetSelf() : nullptr;
  if (!self || !event.target) return;
  u16 shooter = event.weapon.player_id;
  if (shooter == self->id) {
    if (event.target->frequency != self->frequency) {
      last_dealt_tick = GetCurrentTick();
      dealt_yet = true;
    }
    return;
  }
  Player* attacker = bot.game->player_manager.GetPlayerById(shooter);
  Player& target = *event.target;
  if (!attacker || target.id == self->id || attacker->frequency == self->frequency ||
      target.frequency == self->frequency || attacker->frequency == target.frequency || !OnScreen(target)) {
    return;
  }
  enemy_hits[((u32)shooter << 16) | target.id] = GetCurrentTick();
}

void Opportunities::HandleEvent(const WeaponSelfDamageEvent& event) {
  if (bot.game && event.shooter.id == bot.game->player_manager.player_id) return;
  last_hit_tick = GetCurrentTick();
  hit_yet = true;
}

void Opportunities::HandleEvent(const WeaponFireEvent& event) {
  Player* self = bot.game ? bot.game->player_manager.GetSelf() : nullptr;
  const Player& shooter = event.player;
  if (!self || shooter.id == self->id || shooter.frequency == self->frequency || shooter.ship >= 8) return;

  // Bombs only, not mines.
  bool bomb = event.data.type == WeaponType::Bomb || event.data.type == WeaponType::ProximityBomb;
  if (!bomb || event.data.alternate || !OnScreen(shooter)) return;
  if (!Roll(notice) || !Roll(spotting)) return;

  u32 delay = bot.game->connection.settings.ShipSettings[shooter.ship].BombFireDelay;
  window_end[shooter.id] = GetCurrentTick() + delay;
  if (bot.combat_logger) bot.combat_logger->WriteTactic("window", shooter.name);
}

}  // namespace chaos
}  // namespace zero
