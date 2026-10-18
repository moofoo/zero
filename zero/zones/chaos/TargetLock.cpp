#include "TargetLock.h"

#include <zero/CombatLogger.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

// An enemy that deals this fraction of the bot's max energy within kHeavyDamageTicks takes over the lock.
constexpr float kHeavyDamage = 0.3f;
constexpr u32 kHeavyDamageTicks = 300;
// Damage passing within this many ticks counts as the start of a fight, for locking on.
constexpr u32 kContactTicks = 100;
// After having to repel its target's weapons, the bot ignores that enemy for this long.
constexpr u32 kIgnoreTicks = 500;
// The locked enemy's weapons must be at least this share of the danger for a repel to break the lock.
constexpr float kRepelShare = 0.5f;

TargetLock::TargetLock(ZeroBot& bot, float leash_distance, u32 lock_ticks)
    : bot(bot), leash_distance(leash_distance), lock_ticks(lock_ticks) {}

bool TargetLock::Ignored(u16 id, u32 now) const {
  auto it = ignore_until.find(id);
  return it != ignore_until.end() && TICK_GT(it->second, now);
}

int TargetLock::RecentDamage(u16 id, u32 now) {
  auto it = damage_taken.find(id);
  if (it == damage_taken.end()) return 0;

  std::deque<Hit>& hits = it->second;
  while (!hits.empty() && TICK_DIFF(now, hits.front().tick) > (s32)kHeavyDamageTicks) hits.pop_front();

  int total = 0;
  for (const Hit& hit : hits) total += hit.damage;
  return total;
}

void TargetLock::Release(const char* why) {
  locked = kNone;
  reason = why;
}

Player* TargetLock::Choose(const std::vector<Player*>& candidates) {
  Player* self = bot.game->player_manager.GetSelf();
  if (!self) return nullptr;
  u32 now = GetCurrentTick();

  auto find = [&](u16 id) -> Player* {
    for (Player* player : candidates) {
      if (player->id == id) return player;
    }
    return nullptr;
  };

  // Another enemy hitting hard, and harder than the locked one, takes over the lock.
  float heavy = kHeavyDamage * bot.game->ship_controller.ship.energy;
  int locked_damage = locked != kNone ? RecentDamage(locked, now) : 0;
  for (Player* player : candidates) {
    int damage = RecentDamage(player->id, now);
    if (player->id == locked || damage < heavy || damage <= locked_damage) continue;
    if (player->position.Distance(self->position) > leash_distance) continue;

    locked = player->id;
    locked_damage = damage;
    reason = "third party";
  }

  if (locked != kNone) {
    Player* target = find(locked);
    if (!target) {
      Release("gone");
    } else if (target->position.Distance(self->position) > leash_distance) {
      Release("too far");
    } else {
      auto contact = last_contact.find(locked);
      if (contact == last_contact.end() || TICK_DIFF(now, contact->second) > (s32)lock_ticks) Release("quiet");
    }
  }

  // Without a lock, lock onto the enemy within the leash that damage has just passed with.
  if (locked == kNone) {
    u32 latest = 0;
    for (Player* player : candidates) {
      auto contact = last_contact.find(player->id);
      if (contact == last_contact.end() || Ignored(player->id, now)) continue;
      if (TICK_DIFF(now, contact->second) > (s32)kContactTicks) continue;
      if (player->position.Distance(self->position) > leash_distance) continue;

      if (locked == kNone || TICK_GT(contact->second, latest)) {
        locked = player->id;
        latest = contact->second;
        reason = "locked";
      }
    }
  }

  Player* target = locked != kNone ? find(locked) : nullptr;
  if (!target) {
    // The nearest enemy, passing over ones the bot is ignoring unless there's no one else.
    float best = 0.0f;
    for (int pass = 0; pass < 2 && !target; ++pass) {
      for (Player* player : candidates) {
        if (pass == 0 && Ignored(player->id, now)) continue;
        float distance = player->position.DistanceSq(self->position);
        if (!target || distance < best) {
          target = player;
          best = distance;
        }
      }
    }
  }

  u16 id = target ? target->id : kNone;
  bool lock_changed = reason != nullptr && std::string_view(reason) != "nearest";
  if (id != current || lock_changed) {
    if (bot.combat_logger) bot.combat_logger->WriteTarget(target ? target->name : nullptr, reason);
    current = id;
  }
  reason = "nearest";

  return target;
}

void TargetLock::HandleEvent(const WeaponSelfDamageEvent& event) {
  u32 now = GetCurrentTick();
  if (event.shooter.id == bot.game->player_manager.player_id) return;

  last_contact[event.shooter.id] = now;
  damage_taken[event.shooter.id].push_back({now, event.damage});
}

void TargetLock::HandleEvent(const WeaponHitEvent& event) {
  // The bot's own weapon hitting an enemy.
  u16 self_id = bot.game->player_manager.player_id;
  if (event.weapon.player_id != self_id || !event.target || event.target->id == self_id) return;

  last_contact[event.target->id] = GetCurrentTick();
}

void TargetLock::HandleEvent(const WeaponFireEvent& event) {
  // The bot repelling: if mostly its locked target's weapons were coming at it, the fight has gone badly enough to
  // break off.
  u16 self_id = bot.game->player_manager.player_id;
  if (event.player.id != self_id || event.data.type != WeaponType::Repel || locked == kNone) return;

  Game& game = *bot.game;
  float reach = game.connection.settings.RepelDistance / 16.0f;
  float total = 0.0f;
  float from_target = 0.0f;

  for (size_t i = 0; i < game.weapon_manager.weapon_count; ++i) {
    Weapon& weapon = game.weapon_manager.weapons[i];
    if (weapon.frequency == event.player.frequency) continue;
    if (weapon.position.Distance(event.player.position) > reach) continue;

    float damage = (float)GetEstimatedWeaponDamage(weapon, game.connection);
    total += damage;
    if (weapon.player_id == locked) from_target += damage;
  }

  if (total > 0.0f && from_target >= total * kRepelShare) {
    ignore_until[locked] = GetCurrentTick() + kIgnoreTicks;
    Release("repel");
  }
}

}  // namespace chaos
}  // namespace zero
