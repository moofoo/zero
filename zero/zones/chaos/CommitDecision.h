#pragma once

#include <zero/Types.h>

#include <random>

namespace zero {

struct Player;
struct ZeroBot;

namespace chaos {

struct EnergyEstimate;

// What a bot is doing in a fight, chosen by the commit decision:
// - Commit: close in and finish them.
// - Pressure: fight in the bot's own style.
// - Hold: keep its distance and pick shots.
// - Disengage: back off out of the fight.
// - Recharge: stay away until its energy is back.
enum class CombatMode { Commit, Pressure, Hold, Disengage, Recharge };

const char* to_string(CombatMode mode);

struct DecisionSettings {
  // Commit when the chance of winning the exchange is at least this; disengage below the other (aggression).
  float commit_confidence = 0.6f;
  float disengage_confidence = 0.35f;
  // How far off the bot's judgement of that chance is, and how often it decides (skill).
  float judgement_noise = 0.0f;
  u32 decision_ticks = 10;
  // Where a backing-off bot stops, and the fraction of max energy recharging stops at.
  float retreat_distance = 35.0f;
  float recharge_to = 0.9f;
};

// The commit decision (ticket 05): whether to commit, pressure, hold, disengage or recharge, from a quick estimate of
// who would run out of energy first if the fight against the target carried on. The estimate uses only what a player
// in the bot's seat would know: its own state, the target's estimated energy and its uncertainty (EnergyEstimate),
// where the target and other enemies are and which way they face, and the arena's ship settings. Hit chances come from
// measured games. A mode must clear its threshold by a margin to replace the current one.
struct CombatDecision {
  CombatDecision(ZeroBot& bot, EnergyEstimate& estimate, DecisionSettings settings);

  // The mode for this tick, re-deciding when due.
  CombatMode Decide(Player& self, Player& target);

  // The chance of winning behind the last decision, as judged (with the bot's judgement noise).
  float win_chance = 0.5f;

 private:
  ZeroBot& bot;
  EnergyEstimate& estimate;
  DecisionSettings settings;
  std::mt19937 random;

  CombatMode mode = CombatMode::Pressure;
  u32 next_decision_tick = 0;

  // The chance the target runs out of energy before the bot does, if both keep fighting as now.
  float WinChance(Player& self, Player& target);
  // Expected damage per second one player does to another, from what the bot can see.
  float DamageRate(Player& shooter, Player& victim, bool shooter_is_self);
  CombatMode Choose(float chance, Player& self, Player& target);
};

}  // namespace chaos
}  // namespace zero
