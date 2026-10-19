#include "CommitDecision.h"

#include "EnergyEstimate.h"
#include "Opportunities.h"

#include <math.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

// Bullet hit chance by distance, in 5-tile steps, measured over 83 Chaos games on 2026-10-18 (hits over shots fired at
// that distance from the nearest enemy).
constexpr float kBulletHitChance[] = {0.78f, 0.51f, 0.41f, 0.30f, 0.21f, 0.19f, 0.17f, 0.18f,
                                      0.17f, 0.12f, 0.09f, 0.09f, 0.09f, 0.10f, 0.08f};
constexpr float kHitChanceStep = 5.0f;
// Bombs: hits per bomb fired (a bomb can hit more than one ship), and the average hit as a share of a direct hit's
// damage, from the same games (420 of 750).
constexpr float kBombHitsPerBomb = 0.33f;
constexpr float kBombAverageDamage = 0.56f;
// Bots fire only when a shot lines up: about a quarter of their ships' maximum rate while engaged.
constexpr float kEngagedFireFraction = 0.25f;
// The exchange is judged as a fight at no more than this distance (tiles): the question is who would win if it came to
// a fight, not who wins a standoff at long range, where neither side can out-damage the other's recharge.
constexpr float kFightDistance = 15.0f;
// A ship facing away from its victim still turns and fires: the least share of its damage counted.
constexpr float kFacingAwayShare = 0.3f;
// Other enemies further than this (tiles) don't count towards the danger.
constexpr float kThirdPartyDistance = 50.0f;
// When only one side would wear the other down, how sure the bot is of the outcome.
constexpr float kOneSided = 0.8f;
// A new mode must clear its threshold by this much to replace the current one.
constexpr float kHysteresis = 0.05f;
// A commit lasts at least this long (ticks), unless the fight turns clearly against the bot.
constexpr u32 kMinCommitTicks = 150;

const char* to_string(CombatMode mode) {
  switch (mode) {
    case CombatMode::Commit:
      return "commit";
    case CombatMode::Pressure:
      return "pressure";
    case CombatMode::Hold:
      return "hold";
    case CombatMode::Disengage:
      return "disengage";
    case CombatMode::Recharge:
      return "recharge";
    default:
      return "?";
  }
}

float BulletHitChance(float distance) {
  int step = (int)(distance / kHitChanceStep);
  int steps = (int)(sizeof(kBulletHitChance) / sizeof(kBulletHitChance[0]));
  if (step < 0) step = 0;
  return kBulletHitChance[step < steps ? step : steps - 1];
}

float BombExpectedDamage(float direct_hit_damage) {
  return direct_hit_damage * kBombAverageDamage * kBombHitsPerBomb;
}

float BulletDamage(const ArenaSettings& arena, float guns) {
  float level = guns > 1.0f ? guns - 1.0f : 0.0f;
  return (arena.BulletDamageLevel + arena.BulletDamageUpgrade * level) / 1000.0f;
}

float GuessedLevel(u32 initial, u32 maximum) {
  return (initial + maximum) / 2.0f;
}

// Shots per second a ship can fire, from its fire delay in ticks.
static float FireRate(u16 delay_ticks) {
  return delay_ticks > 0 ? 100.0f / delay_ticks : 0.0f;
}

// The standard normal distribution's cumulative probability.
static float NormalCdf(float x) {
  return 0.5f * erfcf(-x / sqrtf(2.0f));
}

CommitDecision::CommitDecision(ZeroBot& bot, EnergyEstimate& estimate, Opportunities& opportunities,
                               DecisionSettings settings)
    : bot(bot), estimate(estimate), opportunities(opportunities), settings(settings), random(std::random_device{}()) {}

float CommitDecision::DamageRate(Player& shooter, Player& victim, bool shooter_is_self) {
  if (shooter.ship >= 8) return 0.0f;
  auto& arena = bot.game->connection.settings;
  auto& ship = arena.ShipSettings[shooter.ship];

  // Gun and bomb levels: the bot knows its own; another ship's are guessed.
  Ship& own = bot.game->ship_controller.ship;
  float guns = shooter_is_self ? (float)own.guns : GuessedLevel(ship.InitialGuns, ship.MaxGuns);
  float bombs = shooter_is_self ? (float)own.bombs : GuessedLevel(ship.InitialBombs, ship.MaxBombs);

  float distance = shooter.position.Distance(victim.position);
  float fight_distance = distance < kFightDistance ? distance : kFightDistance;
  float hit_chance = BulletHitChance(fight_distance);
  float bullet_factor = shooter_is_self ? opportunities.SlowFactor(victim, false, fight_distance) : 1.0f;
  float bomb_factor = shooter_is_self ? opportunities.SlowFactor(victim, true, fight_distance) : 1.0f;

  float rate = 0.0f;
  if (guns > 0.0f) {
    float chance = hit_chance * bullet_factor < 1.0f ? hit_chance * bullet_factor : 1.0f;
    rate += FireRate(ship.BulletFireDelay) * kEngagedFireFraction * chance * BulletDamage(arena, guns);
  }
  if (bombs > 0.0f) {
    rate += FireRate(ship.BombFireDelay) * kEngagedFireFraction *
            BombExpectedDamage(arena.BombDamageLevel / 1000.0f) * bomb_factor;
  }

  // Facing away, they have to turn first.
  Vector2f to_victim = Normalize(victim.position - shooter.position);
  float facing = shooter.GetHeading().Dot(to_victim);
  if (facing < 0.0f) facing = 0.0f;
  return rate * (kFacingAwayShare + (1.0f - kFacingAwayShare) * facing);
}

float CommitDecision::FiringCost(const ShipSettings& ship, float guns, float bombs) {
  float cost = 0.0f;
  if (guns > 0.0f) cost += FireRate(ship.BulletFireDelay) * ship.BulletFireEnergy * guns;
  if (bombs > 0.0f) {
    cost += FireRate(ship.BombFireDelay) * (ship.BombFireEnergy + ship.BombFireEnergyUpgrade * (bombs - 1.0f));
  }
  return cost * kEngagedFireFraction;
}

float CommitDecision::WinChance(Player& self, Player& target, float self_energy) {
  auto& arena = bot.game->connection.settings;
  auto& target_ship = arena.ShipSettings[target.ship];
  Ship& own = bot.game->ship_controller.ship;

  // How fast each side loses energy: the other's damage plus its own firing, less recharge (per second).
  float self_spend = FiringCost(arena.ShipSettings[self.ship], (float)own.guns, (float)own.bombs);
  float self_recharge = own.recharge / 10.0f;
  float incoming = DamageRate(target, self, false);

  // Other enemies near the bot add to the danger.
  auto& players = bot.game->player_manager;
  for (size_t i = 0; i < players.player_count; ++i) {
    Player& other = players.players[i];
    if (other.id == self.id || other.id == target.id || other.ship >= 8 || other.frequency == self.frequency) continue;
    if (other.position.Distance(self.position) > kThirdPartyDistance) continue;
    incoming += DamageRate(other, self, false);
  }

  float target_spend = FiringCost(target_ship, GuessedLevel(target_ship.InitialGuns, target_ship.MaxGuns),
                                  GuessedLevel(target_ship.InitialBombs, target_ship.MaxBombs));
  float target_recharge = GuessedLevel(target_ship.InitialRecharge, target_ship.MaximumRecharge) / 10.0f;
  float outgoing = DamageRate(self, target, true);

  // A fire window: the target can't fire for a moment, which is worth the damage it would have dealt.
  self_energy += DamageRate(target, self, false) * opportunities.WindowSeconds(target);

  float self_loss = incoming + self_spend - self_recharge;
  float target_loss = outgoing + target_spend - target_recharge;

  // When only one side would wear the other down, that side is likely, but not certain, to win.
  if (self_loss <= 0.0f && target_loss <= 0.0f) return 0.5f;
  if (self_loss <= 0.0f) return kOneSided;
  if (target_loss <= 0.0f) return 1.0f - kOneSided;

  // The bot lasts self_energy / self_loss seconds. The target runs out first if its energy is below what it loses in
  // that time; its energy is only an estimate, with its uncertainty.
  float self_time = self_energy / self_loss;
  EnergyEstimate::Reading reading = estimate.Get(target);
  float needed = target_loss * self_time;
  if (reading.sd <= 1.0f) return reading.energy < needed ? kOneSided : 1.0f - kOneSided;
  return NormalCdf((needed - reading.energy) / reading.sd);
}

CombatMode CommitDecision::Choose(float chance, Player& self, Player& target) {
  float commit = settings.commit_confidence + settings.patience;
  float disengage = settings.disengage_confidence + settings.patience;
  float middle = (commit + disengage) / 2.0f;
  float max_energy = (float)bot.game->ship_controller.ship.energy;
  bool can_commit = self.energy >= settings.commit_energy * max_energy;
  u32 now = GetCurrentTick();

  // A commit holds for its minimum time unless the bot would now disengage.
  if (mode == CombatMode::Commit && TICK_DIFF(now, commit_tick) < (s32)kMinCommitTicks && chance >= disengage) {
    return CombatMode::Commit;
  }

  // Recharging lasts until energy is back, unless the fight turns clearly in the bot's favour. Then the bot fights
  // again, in proportion to how it judges the fight (never straight back to disengaging).
  if (mode == CombatMode::Recharge) {
    if (self.energy < settings.recharge_to * max_energy && chance < commit + kHysteresis) return CombatMode::Recharge;
    if (chance >= commit + kHysteresis && can_commit) return CombatMode::Commit;
    return chance >= middle ? CombatMode::Pressure : CombatMode::Hold;
  }

  // Backing off becomes recharging once the bot is out of reach.
  if (mode == CombatMode::Disengage && chance < disengage + kHysteresis &&
      self.position.Distance(target.position) >= settings.retreat_distance * 0.9f) {
    return CombatMode::Recharge;
  }

  float commit_bar = mode == CombatMode::Commit ? commit - kHysteresis : commit + kHysteresis;
  float disengage_bar = mode == CombatMode::Disengage ? disengage + kHysteresis : disengage - kHysteresis;
  if (chance >= commit_bar && can_commit) return CombatMode::Commit;
  if (chance < disengage_bar) return CombatMode::Disengage;

  // Between the two: pressure in the upper half, hold in the lower, with the same margin to switch.
  if (mode == CombatMode::Pressure) return chance >= middle - kHysteresis ? CombatMode::Pressure : CombatMode::Hold;
  if (mode == CombatMode::Hold) return chance >= middle + kHysteresis ? CombatMode::Pressure : CombatMode::Hold;
  return chance >= middle ? CombatMode::Pressure : CombatMode::Hold;
}

float CommitDecision::WinChanceAfter(Player& self, Player& target, float damage) {
  float energy = self.energy - damage;
  return energy > 0.0f ? WinChance(self, target, energy) : 0.0f;
}

CombatMode CommitDecision::Decide(Player& self, Player& target) {
  u32 now = GetCurrentTick();
  if (TICK_GT(next_decision_tick, now)) return mode;
  next_decision_tick = now + settings.decision_ticks;

  float chance = WinChance(self, target, self.energy);
  if (settings.judgement_noise > 0.0f) {
    chance += std::normal_distribution<float>(0.0f, settings.judgement_noise)(random);
  }
  if (chance < 0.0f) chance = 0.0f;
  if (chance > 1.0f) chance = 1.0f;

  win_chance = chance;
  CombatMode chosen = Choose(chance, self, target);
  if (chosen == CombatMode::Commit && mode != CombatMode::Commit) commit_tick = now;
  mode = chosen;
  return mode;
}

}  // namespace chaos
}  // namespace zero
