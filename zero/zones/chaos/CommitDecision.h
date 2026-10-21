#pragma once

#include <zero/Types.h>
#include <zero/game/GameEvent.h>

#include <random>
#include <unordered_map>

namespace zero {

struct ArenaSettings;
struct Player;
struct ShipSettings;
struct ZeroBot;

namespace chaos {

struct EnemySight;
struct EnergyEstimate;
struct Opportunities;
struct WeaponNotice;

// What a bot is doing in a fight, chosen by the commit decision:
// - Commit: close in and finish them.
// - Pressure: fight in the bot's own style.
// - Hold: keep its distance and pick shots.
// - Disengage: back off out of the fight.
// - Recharge: stay away until its energy is back.
// The order matters: it indexes per-mode tables and is how the mode is kept on the blackboard (as an int).
enum class CombatMode { Commit, Pressure, Hold, Disengage, Recharge, Count };

const char* to_string(CombatMode mode);

// A bullet's chance of hitting an enemy this far away (tiles), measured from Chaos games.
float BulletHitChance(float distance);
// A bomb's expected damage per bomb fired at an enemy this far away (tiles), as a share of a direct hit's damage,
// measured from Chaos games: blast and shrapnel, for a bot leading its shots by this much (LeadSkill, 0-1). Better
// leading keeps more of a long bomb's chance.
float BombDamageShare(float distance, float lead_skill);
// The damage a bomb fired at an enemy this far away (tiles) does to its own shooter, on average, as a share of a direct
// hit's damage: measured from Chaos games (ticket 28), interpolated between distances.
float BombSelfDamageShare(float distance);
// A bullet's damage at a gun level (1 and up), from the arena's settings.
float BulletDamage(const ArenaSettings& arena, float guns);
// A ship's gun or bomb level as another player guesses it: the middle of its initial-to-maximum range.
float GuessedLevel(u32 initial, u32 maximum);

struct DecisionSettings {
  // Commit when the chance of winning the exchange is at least this; disengage below the other (aggression).
  float commit_confidence = 0.6f;
  float disengage_confidence = 0.35f;
  // Never commit with less than this fraction of max energy (aggression).
  float commit_energy = 0.5f;
  // Added to both confidences: a patient bot needs better odds to commit and backs off sooner (from Judgement).
  float patience = 0.0f;
  // How far off the bot's judgement of that chance is, and how often it decides, in ticks (from Judgement).
  float judgement_noise = 0.0f;
  u32 decision_ticks = 10;
  // Where a backing-off bot stops (tiles), and the fraction of max energy recharging stops at.
  float retreat_distance = 35.0f;
  float recharge_to = 0.9f;
  // The bot's LeadSkill, for its own bombs' expected damage (BombDamageShare).
  float lead_skill = 1.0f;
};

// The commit decision: whether to commit, pressure, hold, disengage or recharge, from a quick estimate of who would run
// out of energy first if the bot fought the target at fighting range. The estimate uses only what a player in the
// bot's seat would know: its own state, the target's estimated energy and its uncertainty (EnergyEstimate), where the
// target and other enemies are and which way they face, and the arena's ship settings. Hit chances come from measured
// games. The opponent's mistakes the bot has spotted count too (Opportunities): a fire window is energy in the bot's
// favour, the damage the target can't deal during it, and a slow target takes more of the bot's damage. A mode must
// clear its threshold by a margin to replace the current one.
// A won bomb exchange (ticket 28) re-decides at once rather than when the next decision is due: the bot's bomb hits an
// enemy straight on (within kWonBlast tiles of the blast) while that enemy's last bomb it noticed, in the kWonTicks
// before, missed it. A player takes that as the cue to close in.
struct CommitDecision : EventHandler<WeaponHitEvent>, EventHandler<WeaponSelfDamageEvent> {
  CommitDecision(ZeroBot& bot, EnergyEstimate& estimate, Opportunities& opportunities, EnemySight& sight,
                 WeaponNotice* notice, DecisionSettings settings);

  void HandleEvent(const WeaponHitEvent& event) override;
  void HandleEvent(const WeaponSelfDamageEvent& event) override;

  // The mode for this tick, re-deciding when due.
  CombatMode Decide(Player& self, Player& target);

  // The chance of winning if the bot first takes this much damage (for accepting a hit).
  float WinChanceAfter(Player& self, Player& target, float damage);

  // The mode from the last decision.
  CombatMode Mode() const { return mode; }

  // The chance of winning behind the last decision, as judged (with the bot's judgement noise).
  float win_chance = 0.5f;

 private:
  ZeroBot& bot;
  EnergyEstimate& estimate;
  Opportunities& opportunities;
  EnemySight& sight;
  WeaponNotice* notice;
  DecisionSettings settings;
  // Each other ship's facing toward the bot, averaged over time, and when it was last updated.
  struct Average {
    float value;
    u32 tick;
  };
  std::unordered_map<u16, Average> facing_average;
  // When each enemy's bomb last hurt the bot, to tell a missed bomb from one that landed.
  std::unordered_map<u16, u32> bomb_damage_tick;
  std::mt19937 random;

  CombatMode mode = CombatMode::Pressure;
  u32 next_decision_tick = 0;
  // When the bot last committed: a commit lasts a minimum time.
  u32 commit_tick = 0;

  // The chance the target runs out of energy before the bot does, if both fight on at fighting range, with the bot
  // at this energy.
  float WinChance(Player& self, Player& target, float self_energy);
  // Expected damage per second one player does to another at up to fighting range, from what the bot can see. The
  // bot's own damage counts a slow target it has spotted.
  float DamageRate(Player& shooter, Player& victim, bool shooter_is_self);
  // Energy per second a ship spends firing while engaged.
  float FiringCost(const ShipSettings& ship, float guns, float bombs);
  // Another ship's facing toward the bot (0-1), averaged over the last moments, with this facing now.
  float AverageFacing(u16 id, float facing);
  // The energy a ship loses to its own bombs while bombing an enemy this far away (per second).
  float OwnBombDamageRate(const ShipSettings& ship, float bombs, float distance);
  CombatMode Choose(float chance, Player& self, Player& target);
};

}  // namespace chaos
}  // namespace zero
