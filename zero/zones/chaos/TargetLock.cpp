#include "TargetLock.h"

#include "EnergyEstimate.h"
#include "Opportunities.h"

#include <math.h>

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
// The locked enemy's weapons must be more than this share of the danger for a repel to break the lock.
constexpr float kRepelShare = 0.5f;
// Battle blasting: the bot must have been out of any fight this long (seconds) and have this share of its max energy;
// a battle lasts at most kMaxBattleTicks, and a pair not seen fighting for kBattleGapTicks is a new opportunity.
constexpr float kBattleQuietSeconds = 3.0f;
constexpr float kBattleEnergy = 0.7f;
constexpr u32 kMaxBattleTicks = 1000;
constexpr u32 kBattleGapTicks = 300;

TargetLock::TargetLock(ZeroBot& bot, float leash_distance, u32 lock_ticks, EnergyEstimate& estimate,
                       Vector2f view_extent, std::optional<float> weak_enemy_energy, Opportunities& opportunities)
    : bot(bot),
      leash_distance(leash_distance),
      lock_ticks(lock_ticks),
      estimate(estimate),
      view_extent(view_extent),
      weak_enemy_energy(weak_enemy_energy),
      opportunities(opportunities) {}

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

void TargetLock::Note(const char* why) {
  if (!reason.empty()) reason += ", ";
  reason += why;
}

void TargetLock::Release(const char* why) {
  locked = kNone;
  Note(why);
}

bool TargetLock::OnScreen(const Player& player, const Player& self) const {
  Vector2f delta = player.position - self.position;
  return fabsf(delta.x) <= view_extent.x && fabsf(delta.y) <= view_extent.y;
}

bool TargetLock::WithinLeash(const Player& player, const Player& self) const {
  return player.position.Distance(self.position) <= leash_distance;
}

void TargetLock::UpdateBattle(const std::vector<Player*>& candidates, const Player& self, u32 now) {
  auto find = [&](u16 id) -> Player* {
    for (Player* player : candidates) {
      if (player->id == id) return player;
    }
    return nullptr;
  };

  if (battling) {
    auto hit = opportunities.LastHitTick();
    bool hit_since = hit && !TICK_GT(battle_start, *hit);
    if (!opportunities.BattleGoing(battle_a, battle_b) || hit_since || TICK_GT(now, battle_end) || !find(locked)) {
      battling = false;
      Note("battle over");
    } else {
      last_contact[locked] = now;
    }
    return;
  }

  float max_energy = (float)bot.game->ship_controller.ship.energy;
  if (opportunities.SecondsSinceFight() < kBattleQuietSeconds || self.energy < kBattleEnergy * max_energy) return;
  auto battle = opportunities.FindBattle(candidates, self);
  if (!battle) return;

  u32 key = PairKey(battle->first, battle->second);
  auto seen = battle_seen.find(key);
  bool fresh = seen == battle_seen.end() || TICK_DIFF(now, seen->second) > (s32)kBattleGapTicks;
  battle_seen[key] = now;
  Player* a = find(battle->first);
  Player* b = find(battle->second);
  if (!fresh || !a || !b || !opportunities.Spot()) return;

  Player* weaker = estimate.Get(*a).energy <= estimate.Get(*b).energy ? a : b;
  locked = weaker->id;
  battling = true;
  battle_a = a->id;
  battle_b = b->id;
  battle_start = now;
  battle_end = now + kMaxBattleTicks;
  last_contact[locked] = now;
  Note("battle");
  if (bot.combat_logger) bot.combat_logger->WriteTactic("battle", weaker->name);
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

  // Whether the locked enemy is a weak one, by the bot's estimate.
  auto locked_weak = [&]() {
    Player* target = locked != kNone ? find(locked) : nullptr;
    return weak_enemy_energy && target && estimate.Get(*target).energy < *weak_enemy_energy;
  };

  UpdateBattle(candidates, *self, now);

  // A bot that goes after weak enemies switches to the weakest one on its screen below its threshold, by its estimate,
  // unless its target is weak already.
  if (weak_enemy_energy && !battling) {
    if (!locked_weak()) {
      Player* weakest = nullptr;
      float weakest_energy = *weak_enemy_energy;
      for (Player* player : candidates) {
        if (Ignored(player->id, now) || !OnScreen(*player, *self) || !WithinLeash(*player, *self)) continue;
        float energy = estimate.Get(*player).energy;
        if (energy < weakest_energy) {
          weakest = player;
          weakest_energy = energy;
        }
      }
      if (weakest && weakest->id != locked) {
        locked = weakest->id;
        // Going after them starts the fight, so the quiet timer runs from now.
        last_contact[locked] = now;
        Note("weak enemy");
      }
    }
  }

  // Another enemy hitting hard, and harder than the locked one, takes over the lock (ship.energy is max energy), unless
  // the bot is going after a weak enemy.
  float heavy = kHeavyDamage * bot.game->ship_controller.ship.energy;
  int locked_damage = locked != kNone ? RecentDamage(locked, now) : 0;
  u16 taker = kNone;
  for (Player* player : candidates) {
    if (battling || locked_weak()) break;
    int damage = RecentDamage(player->id, now);
    if (player->id == locked || damage < heavy || damage <= locked_damage) continue;
    if (Ignored(player->id, now) || !WithinLeash(*player, *self)) continue;

    taker = player->id;
    locked_damage = damage;
  }
  if (taker != kNone) {
    locked = taker;
    Note("third party");
  }

  if (locked != kNone && !battling) {
    Player* target = find(locked);
    if (!target) {
      Release("gone");
    } else if (!WithinLeash(*target, *self)) {
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
      if (TICK_DIFF(now, contact->second) > (s32)kContactTicks || !WithinLeash(*player, *self)) continue;

      if (locked == kNone || TICK_GT(contact->second, latest)) {
        locked = player->id;
        latest = contact->second;
      }
    }
    if (locked != kNone) Note("locked");
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

  // Log a new target, or a change of lock on the same one, with everything that led to it.
  u16 id = target ? target->id : kNone;
  if (id != current || !reason.empty()) {
    if (bot.combat_logger) {
      bot.combat_logger->WriteTarget(target ? target->name : nullptr, reason.empty() ? "nearest" : reason.c_str());
    }
    current = id;
  }
  reason.clear();

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

  if (total > 0.0f && from_target > total * kRepelShare) {
    ignore_until[locked] = GetCurrentTick() + kIgnoreTicks;
    Release("repel");
  }
}

}  // namespace chaos
}  // namespace zero
