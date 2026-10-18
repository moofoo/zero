#include "CommitDecision.h"

#include "EnergyEstimate.h"

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
// A ship facing away from its victim still turns and fires: the least share of its damage counted.
constexpr float kFacingAwayShare = 0.3f;
// Other enemies further than this (tiles) don't count towards the danger.
constexpr float kThirdPartyDistance = 50.0f;
// A new mode must clear its threshold by this much to replace the current one.
constexpr float kHysteresis = 0.05f;

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
  }
  return "?";
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

// The standard normal distribution's cumulative probability.
static float NormalCdf(float x) {
  return 0.5f * erfcf(-x / sqrtf(2.0f));
}

CombatDecision::CombatDecision(ZeroBot& bot, EnergyEstimate& estimate, DecisionSettings settings)
    : bot(bot), estimate(estimate), settings(settings), random(std::random_device{}()) {}

float CombatDecision::DamageRate(Player& shooter, Player& victim, bool shooter_is_self) {
  if (shooter.ship >= 8) return 0.0f;
  auto& arena = bot.game->connection.settings;
  auto& ship = arena.ShipSettings[shooter.ship];

  // Gun and bomb levels: the bot knows its own; another ship's are guessed at the middle of its range.
  Ship& own = bot.game->ship_controller.ship;
  float guns = shooter_is_self ? (float)own.guns : (ship.InitialGuns + ship.MaxGuns) / 2.0f;
  float bombs = shooter_is_self ? (float)own.bombs : (ship.InitialBombs + ship.MaxBombs) / 2.0f;

  float hit_chance = BulletHitChance(shooter.position.Distance(victim.position));

  float rate = 0.0f;
  if (guns > 0.0f && ship.BulletFireDelay > 0) {
    float damage = (arena.BulletDamageLevel + arena.BulletDamageUpgrade * (guns - 1.0f)) / 1000.0f;
    rate += 100.0f / ship.BulletFireDelay * kEngagedFireFraction * hit_chance * damage;
  }
  if (bombs > 0.0f && ship.BombFireDelay > 0) {
    rate += 100.0f / ship.BombFireDelay * kEngagedFireFraction * BombExpectedDamage(arena.BombDamageLevel / 1000.0f);
  }

  // Facing away, they have to turn first.
  Vector2f to_victim = Normalize(victim.position - shooter.position);
  float facing = shooter.GetHeading().Dot(to_victim);
  if (facing < 0.0f) facing = 0.0f;
  return rate * (kFacingAwayShare + (1.0f - kFacingAwayShare) * facing);
}

float CombatDecision::WinChance(Player& self, Player& target) {
  auto& arena = bot.game->connection.settings;
  auto& self_ship = arena.ShipSettings[self.ship];
  auto& target_ship = arena.ShipSettings[target.ship];

  // How fast each side loses energy: the other's damage plus its own firing, less recharge (per second).
  float bullet_cost = self_ship.BulletFireEnergy * (float)bot.game->ship_controller.ship.guns;
  float self_fire_rate = 100.0f / (self_ship.BulletFireDelay ? self_ship.BulletFireDelay : 100);
  float self_spend = self_fire_rate * kEngagedFireFraction * bullet_cost;
  float self_recharge = bot.game->ship_controller.ship.recharge / 10.0f;
  float incoming = DamageRate(target, self, false);

  // Other enemies near the bot add to the danger.
  auto& players = bot.game->player_manager;
  for (size_t i = 0; i < players.player_count; ++i) {
    Player& other = players.players[i];
    if (other.id == self.id || other.id == target.id || other.ship >= 8 || other.frequency == self.frequency) continue;
    if (other.position.Distance(self.position) > kThirdPartyDistance) continue;
    incoming += DamageRate(other, self, false);
  }

  float target_spend = target_ship.BulletFireEnergy * (target_ship.InitialGuns + target_ship.MaxGuns) / 2.0f * 100.0f /
                       (target_ship.BulletFireDelay ? target_ship.BulletFireDelay : 100) * kEngagedFireFraction;
  float target_recharge = (target_ship.InitialRecharge + target_ship.MaximumRecharge) / 2.0f / 10.0f;
  float outgoing = DamageRate(self, target, true);

  float self_loss = incoming + self_spend - self_recharge;
  float target_loss = outgoing + target_spend - target_recharge;

  // Neither side is losing: an even chance. Only the target losing: a win. Only the bot losing: a loss.
  if (self_loss <= 0.0f && target_loss <= 0.0f) return 0.5f;
  if (self_loss <= 0.0f) return 1.0f;
  if (target_loss <= 0.0f) return 0.0f;

  // The bot lasts self_energy / self_loss seconds. The target runs out first if its energy is below what it loses in
  // that time; its energy is only an estimate, with its uncertainty.
  float self_time = self.energy / self_loss;
  EnergyEstimate::Reading reading = estimate.Get(target);
  float needed = target_loss * self_time;
  if (reading.sd <= 1.0f) return reading.energy < needed ? 1.0f : 0.0f;
  return NormalCdf((needed - reading.energy) / reading.sd);
}

CombatMode CombatDecision::Choose(float chance, Player& self, Player& target) {
  float commit = settings.commit_confidence;
  float disengage = settings.disengage_confidence;

  // Recharging lasts until energy is back, unless the fight turns clearly in the bot's favour.
  if (mode == CombatMode::Recharge) {
    float max_energy = (float)bot.game->ship_controller.ship.energy;
    if (self.energy < settings.recharge_to * max_energy && chance < commit + kHysteresis) return CombatMode::Recharge;
  }

  // Backing off becomes recharging once the bot is out of reach.
  if (mode == CombatMode::Disengage && chance < disengage + kHysteresis &&
      self.position.Distance(target.position) >= settings.retreat_distance * 0.9f) {
    return CombatMode::Recharge;
  }

  float commit_bar = mode == CombatMode::Commit ? commit - kHysteresis : commit + kHysteresis;
  float disengage_bar = mode == CombatMode::Disengage ? disengage + kHysteresis : disengage - kHysteresis;
  if (chance >= commit_bar) return CombatMode::Commit;
  if (chance < disengage_bar) return CombatMode::Disengage;
  // Between the two: pressure in the upper half, hold in the lower, with the same margin to switch.
  float middle = (commit + disengage) / 2.0f;
  if (mode == CombatMode::Pressure) return chance >= middle - kHysteresis ? CombatMode::Pressure : CombatMode::Hold;
  if (mode == CombatMode::Hold) return chance >= middle + kHysteresis ? CombatMode::Pressure : CombatMode::Hold;
  return chance >= middle ? CombatMode::Pressure : CombatMode::Hold;
}

CombatMode CombatDecision::Decide(Player& self, Player& target) {
  u32 now = GetCurrentTick();
  if (TICK_GT(next_decision_tick, now)) return mode;
  next_decision_tick = now + settings.decision_ticks;

  float chance = WinChance(self, target);
  if (settings.judgement_noise > 0.0f) {
    chance += std::normal_distribution<float>(0.0f, settings.judgement_noise)(random);
  }
  if (chance < 0.0f) chance = 0.0f;
  if (chance > 1.0f) chance = 1.0f;

  win_chance = chance;
  mode = Choose(chance, self, target);
  return mode;
}

}  // namespace chaos
}  // namespace zero
