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
