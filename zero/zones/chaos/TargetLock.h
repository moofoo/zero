#pragma once

#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/game/GameEvent.h>

#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace zero {

struct Player;
struct ZeroBot;

namespace chaos {

struct EnergyEstimate;
struct Opportunities;

// Which enemy a Chaos bot fights. Like a player, once damage passes between the bot and an enemy (either way), the
// bot locks onto that enemy and keeps after it even if another comes closer, including after being repelled away.
// It lets go when the enemy:
// - is gone (dead, respawning, spectating, in a safe zone, unreachable)
// - is further away than the leash distance
// - has traded no damage with it for the lock time
// - or when another enemy hits it hard (30% of its max energy within 3 seconds), which takes over the lock
// - or when it had to repel mostly the locked enemy's weapons: then it picks another target and ignores that enemy
//   for a few seconds
// - or, for a bot that goes after weak enemies, when an enemy on its screen is estimated below its weak-enemy
//   threshold (and its target isn't): it switches to the weakest such enemy. Whether to chase it is the commit
//   decision's.
// - or, battle blasting (ticket 24): when the bot isn't fighting and two other enemies are trading hits on its screen
//   at range, it may lock onto the weaker of them (by its spotting chance) and bomb their fight from a distance, until
//   their fight ends, the bot is hit, or a few seconds pass; nothing else moves the lock meanwhile.
// Without a lock, the target is the nearest enemy.
struct TargetLock : EventHandler<WeaponSelfDamageEvent>, EventHandler<WeaponHitEvent>, EventHandler<WeaponFireEvent> {
  static constexpr u16 kNone = 0xFFFF;

  // The estimate and view are what the bot knows of enemy energy and what's on its screen; weak_enemy_energy is unset
  // for a bot that doesn't go after weak enemies.
  TargetLock(ZeroBot& bot, float leash_distance, u32 lock_ticks, EnergyEstimate& estimate, Vector2f view_extent,
             std::optional<float> weak_enemy_energy, Opportunities& opportunities);

  // This tick's target among the enemies the bot could fight, or null when there are none.
  Player* Choose(const std::vector<Player*>& candidates);

  // Whether the bot is bombing two other enemies' fight.
  bool Battling() const { return battling; }

  void HandleEvent(const WeaponSelfDamageEvent& event) override;
  void HandleEvent(const WeaponHitEvent& event) override;
  void HandleEvent(const WeaponFireEvent& event) override;

 private:
  struct Hit {
    u32 tick;
    int damage;
  };

  ZeroBot& bot;
  float leash_distance;
  u32 lock_ticks;
  EnergyEstimate& estimate;
  Vector2f view_extent;
  std::optional<float> weak_enemy_energy;
  Opportunities& opportunities;

  // The locked enemy, and this tick's target (often the same, but without a lock the nearest enemy).
  u16 locked = kNone;
  u16 current = kNone;
  // What happened to the lock this tick, for the log ("too far, locked"); empty if nothing.
  std::string reason;

  // When damage last passed between the bot and each player, either way.
  std::unordered_map<u16, u32> last_contact;
  // Damage each player did to the bot recently.
  std::unordered_map<u16, std::deque<Hit>> damage_taken;
  // Players the bot won't lock onto or target until a tick, after it had to repel them.
  std::unordered_map<u16, u32> ignore_until;
  // Battle blasting: the two enemies fighting, when it started and must end; and when each pair of enemies was last
  // seen fighting, so each fight is an opportunity once.
  bool battling = false;
  u16 battle_a = kNone;
  u16 battle_b = kNone;
  u32 battle_start = 0;
  u32 battle_end = 0;
  std::unordered_map<u32, u32> battle_seen;

  // Adds to this tick's reason for the log.
  void Note(const char* why);
  // Lets go of the locked enemy, for this reason.
  void Release(const char* why);
  // Whether the bot is ignoring this player after having to repel them.
  bool Ignored(u16 id, u32 now) const;
  // Whether a player is close enough to lock onto or stay locked on.
  bool WithinLeash(const Player& player, const Player& self) const;
  // Whether a player is on the bot's screen.
  bool OnScreen(const Player& player, const Player& self) const;
  // Damage this player did to the bot in the last few seconds.
  int RecentDamage(u16 id, u32 now);
  // Starts, keeps or ends battle blasting, moving the lock when a battle starts.
  void UpdateBattle(const std::vector<Player*>& candidates, const Player& self, u32 now);
};

}  // namespace chaos
}  // namespace zero
