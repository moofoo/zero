#include "Opportunities.h"

#include <math.h>

#include <zero/CombatLogger.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

constexpr float kTicksPerSecond = 100.0f;
// How much likelier a shot is to hit a slow target, measured from Chaos games on 2026-10-19 (hits over shots, slow
// targets against all others).
constexpr float kSlowBulletFactor = 1.45f;
constexpr float kSlowBombNearFactor = 1.2f;
constexpr float kSlowBombFarFactor = 1.1f;
constexpr float kSlowBombNearDistance = 20.0f;
// A target not looked at for this long starts a fresh slowdown when next seen slow.
constexpr u32 kSlowForgetTicks = 50;
// Battle blasting: two enemies count as fighting when they're within kBattleSpread tiles of each other and each hit the
// other within kBattleTradeTicks; the nearer must be kBattleNear to kBattleFar tiles from the bot. The fight is over
// when neither has hit the other for kBattleQuietTicks.
constexpr float kBattleSpread = 20.0f;
constexpr float kBattleNear = 30.0f;
constexpr float kBattleFar = 70.0f;
constexpr u32 kBattleTradeTicks = 200;
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

bool Opportunities::Recent(std::optional<u32> tick, u32 ticks) {
  return tick && TICK_DIFF(GetCurrentTick(), *tick) <= (s32)ticks;
}

float Opportunities::WindowSeconds(const Player& player) {
  auto it = window_end.find(player.id);
  if (it == window_end.end()) return 0.0f;
  s32 left = TICK_DIFF(it->second, GetCurrentTick());
  return left > 0 ? left / kTicksPerSecond : 0.0f;
}

bool Opportunities::SpottedSlow(const Player& player) {
  u32 now = GetCurrentTick();
  Slowdown& slowdown = slowdowns[player.id];
  if (TICK_DIFF(now, slowdown.seen_tick) > (s32)kSlowForgetTicks) slowdown.slow = false;
  slowdown.seen_tick = now;

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

float Opportunities::SecondsSinceFight() const {
  std::optional<u32> latest = last_hit_tick;
  if (last_dealt_tick && (!latest || TICK_GT(*last_dealt_tick, *latest))) latest = last_dealt_tick;
  if (!latest) return 1e6f;
  return TICK_DIFF(GetCurrentTick(), *latest) / kTicksPerSecond;
}

std::optional<u32> Opportunities::LastHit(u16 shooter, u16 target) const {
  auto it = enemy_hits.find(PairKey(shooter, target));
  if (it == enemy_hits.end()) return {};
  return it->second;
}

bool Opportunities::BattleGoing(u16 a, u16 b) const {
  return Recent(LastHit(a, b), kBattleQuietTicks) || Recent(LastHit(b, a), kBattleQuietTicks);
}

std::optional<std::pair<u16, u16>> Opportunities::FindBattle(const std::vector<Player*>& candidates,
                                                             const Player& self) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      Player& a = *candidates[i];
      Player& b = *candidates[j];
      if (!Recent(LastHit(a.id, b.id), kBattleTradeTicks) || !Recent(LastHit(b.id, a.id), kBattleTradeTicks)) continue;
      if (a.position.Distance(b.position) > kBattleSpread) continue;
      float da = a.position.Distance(self.position);
      float db = b.position.Distance(self.position);
      float nearer = da < db ? da : db;
      if (nearer < kBattleNear || nearer > kBattleFar) continue;
      return a.id < b.id ? std::make_pair(a.id, b.id) : std::make_pair(b.id, a.id);
    }
  }
  return {};
}

bool Opportunities::FightingOthers(u16 id, u32 ticks) const {
  u32 now = GetCurrentTick();
  for (const auto& [key, tick] : enemy_hits) {
    u16 shooter = (u16)(key >> 16);
    u16 target = (u16)(key & 0xFFFF);
    if ((shooter == id || target == id) && shooter != target && TICK_DIFF(now, tick) <= (s32)ticks) return true;
  }
  return false;
}

void Opportunities::HandleEvent(const WeaponHitEvent& event) {
  Player* self = bot.game ? bot.game->player_manager.GetSelf() : nullptr;
  if (!self || !event.target) return;
  u16 shooter = event.weapon.player_id;
  if (shooter == self->id) {
    if (event.target->frequency != self->frequency) last_dealt_tick = GetCurrentTick();
    return;
  }
  // Another enemy hitting another, noticed on the bot's screen as any weapon is.
  Player* attacker = bot.game->player_manager.GetPlayerById(shooter);
  Player& target = *event.target;
  if (!attacker || target.id == self->id || attacker->frequency == self->frequency ||
      target.frequency == self->frequency || attacker->frequency == target.frequency || !OnScreen(target)) {
    return;
  }
  if (!Roll(notice)) return;
  enemy_hits[PairKey(shooter, target.id)] = GetCurrentTick();
}

void Opportunities::HandleEvent(const WeaponSelfDamageEvent& event) {
  Player* self = bot.game ? bot.game->player_manager.GetSelf() : nullptr;
  if (!self || event.shooter.id == self->id || event.shooter.frequency == self->frequency) return;
  last_hit_tick = GetCurrentTick();
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
