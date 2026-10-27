#include "CommitDecision.h"
#include "ChaosRandom.h"

#include "EnemySight.h"
#include "EnergyEstimate.h"
#include "Motives.h"
#include "Opportunities.h"
#include "WeaponNotice.h"

#include <math.h>
#include <zero/CombatLogger.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>
#include <zero/game/WeaponManager.h>

#include <algorithm>

namespace zero {
namespace chaos {

// Bullet hit chance by distance, in 5-tile steps, measured over 83 Chaos games on 2026-10-18 (hits over shots fired at
// that distance from the nearest enemy).
constexpr float kBulletHitChance[] = {0.78f, 0.51f, 0.41f, 0.30f, 0.21f, 0.19f, 0.17f, 0.18f,
                                      0.17f, 0.12f, 0.09f, 0.09f, 0.09f, 0.10f, 0.08f};
// The distance steps the measured tables are in (tiles).
constexpr float kDistanceStep = 5.0f;
// Bombs: damage per bomb fired (blast and shrapnel, every ship hit) as a share of a direct hit's (750), by distance at
// firing in the same steps, measured on 2026-10-20 over about 100,000 bombs (chaos-offline docs/metrics/bomb-value-
// 2026-10-20.md), for novices and veterans. Veterans keep more at range, and a little more up close.
constexpr float kBombDamageNovice[] = {0.312f, 0.312f, 0.299f, 0.283f, 0.259f, 0.227f, 0.196f, 0.169f,
                                       0.160f, 0.148f, 0.119f, 0.119f, 0.099f, 0.084f, 0.061f};
constexpr float kBombDamageVeteran[] = {0.301f, 0.387f, 0.357f, 0.328f, 0.297f, 0.256f, 0.232f, 0.207f,
                                        0.196f, 0.172f, 0.171f, 0.137f, 0.119f, 0.119f, 0.087f};
static_assert(sizeof(kBombDamageNovice) == sizeof(kBombDamageVeteran), "the bomb tables need the same steps");
// Self-damage per bomb fired, in the same steps and shares, from about 32,000 bot bombs (runs S2, T1b, L1, L2): a
// shooter's own blast as the bomb goes off near it, at an enemy that closed in or against a wall.
constexpr float kBombSelfDamage[] = {0.775f, 0.371f, 0.193f, 0.103f, 0.062f, 0.042f, 0.034f, 0.029f,
                                     0.020f, 0.017f, 0.018f, 0.018f, 0.018f, 0.018f, 0.018f};
// The novices' LeadSkill when the bomb table was measured (the veterans' was 1).
constexpr float kNoviceLeadSkill = 0.3f;
// An enemy's lead skill as the bot guesses it, for the bombs it fires: between a novice's and a veteran's.
constexpr float kGuessedLeadSkill = 0.65f;
// Bots fire only when a shot lines up: about a quarter of their ships' maximum rate while engaged.
constexpr float kEngagedFireFraction = 0.25f;
// The exchange is judged as a fight at no more than this distance (tiles): the question is who would win if it came to
// a fight, not who wins a standoff at long range, where neither side can out-damage the other's recharge.
constexpr float kFightDistance = 15.0f;
// A ship facing away from its victim still turns and fires: the least share of its damage counted.
constexpr float kFacingAwayShare = 0.3f;
// Another ship's facing is judged over about this long (ticks), not at an instant: one that has been flying away isn't
// fighting, but one that has just turned is.
constexpr float kFacingMemoryTicks = 200.0f;
// Other enemies further than this (tiles) don't count towards the danger.
constexpr float kThirdPartyDistance = 50.0f;
// When only one side would wear the other down, how sure the bot is of the outcome.
constexpr float kOneSided = 0.8f;
// A new mode must clear its threshold by this much to replace the current one.
constexpr float kHysteresis = 0.05f;
// A commit lasts at least this long (ticks), unless the fight turns clearly against the bot.
constexpr u32 kMinCommitTicks = 150;
// A won bomb exchange: the bot's bomb within this many tiles of the target as it goes off, and the target's last bomb
// at the bot, missed, within this many ticks before.
constexpr float kWonBlast = 1.5f;
// A commit begun further away than this (tiles) is a closing-in: it holds unless the odds fall below the disengage
// threshold, and on arriving within this distance the fight is judged afresh, staying committed only if the odds
// still clear the commit threshold. A commit up close also holds unless the odds fall below the disengage threshold:
// small drifts don't break a finish.
constexpr float kCommitRange = 25.0f;
// Against the enemy it holds a grudge against, the bot's commit and disengage thresholds are this much lower. Zero: a
// grudge bot hunts its killer but fights it with its usual judgement (with 0.1 it died twice as often as a roamer and
// settled few more grudges, ticket 09).
constexpr float kGrudgeAppetite = 0.0f;
// A stalking Spider's strike holds the commit this long (ticks), unless its own energy falls below kStrikeAbortEnergy
// of max: time to close in and fire (ticket 29).
constexpr u32 kStrikeTicks = 500;
constexpr float kStrikeAbortEnergy = 0.3f;
constexpr u32 kWonTicks = 300;

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

// The step of a measured table for a distance, clamped to the table.
static int DistanceStep(float distance, int steps) {
  int step = (int)(distance / kDistanceStep);
  return step < 0 ? 0 : step >= steps ? steps - 1 : step;
}

float BulletHitChance(float distance) {
  return kBulletHitChance[DistanceStep(distance, (int)(sizeof(kBulletHitChance) / sizeof(kBulletHitChance[0])))];
}

float BombSelfDamageShare(float distance) {
  // Each step's value stands for the middle of its distances; between middles, a straight line.
  constexpr int steps = (int)(sizeof(kBombSelfDamage) / sizeof(kBombSelfDamage[0]));
  float position = distance / kDistanceStep - 0.5f;
  if (position <= 0.0f) return kBombSelfDamage[0];
  int step = (int)position;
  if (step >= steps - 1) return kBombSelfDamage[steps - 1];
  float t = position - (float)step;
  return kBombSelfDamage[step] + (kBombSelfDamage[step + 1] - kBombSelfDamage[step]) * t;
}

float BombDamageShare(float distance, float lead_skill) {
  int step = DistanceStep(distance, (int)(sizeof(kBombDamageVeteran) / sizeof(kBombDamageVeteran[0])));
  float t = (lead_skill - kNoviceLeadSkill) / (1.0f - kNoviceLeadSkill);
  t = t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t;
  return kBombDamageNovice[step] + (kBombDamageVeteran[step] - kBombDamageNovice[step]) * t;
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

CommitDecision::CommitDecision(ZeroBot& bot, EnergyEstimate& estimate, Opportunities& opportunities, EnemySight& sight,
                               WeaponNotice* notice, DecisionSettings settings)
    : bot(bot),
      estimate(estimate),
      opportunities(opportunities),
      sight(sight),
      notice(notice),
      settings(settings),
      random(ChaosSeed()) {}

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
  float bullet_factor = shooter_is_self ? opportunities.SlowFactor(victim, false, distance) : 1.0f;
  float bomb_factor = shooter_is_self ? opportunities.SlowFactor(victim, true, distance) : 1.0f;

  float rate = 0.0f;
  if (guns > 0.0f) {
    float chance = hit_chance * bullet_factor < 1.0f ? hit_chance * bullet_factor : 1.0f;
    rate += FireRate(ship.BulletFireDelay) * kEngagedFireFraction * chance * BulletDamage(arena, guns);
  }
  if (bombs > 0.0f) {
    float lead = shooter_is_self ? settings.lead_skill : kGuessedLeadSkill;
    rate += FireRate(ship.BombFireDelay) * kEngagedFireFraction *
            BombDamageShare(fight_distance, lead) * (arena.BombDamageLevel / 1000.0f) * bomb_factor;
  }

  // Facing away, they have to turn first. The bot's own damage doesn't depend on its facing: it turns to shoot in its
  // aim phases. Another ship's is its facing over the last moments.
  if (shooter_is_self) return rate;
  Vector2f to_victim = Normalize(victim.position - shooter.position);
  float facing = shooter.GetHeading().Dot(to_victim);
  if (facing < 0.0f) facing = 0.0f;
  return rate * (kFacingAwayShare + (1.0f - kFacingAwayShare) * AverageFacing(shooter.id, facing));
}

float CommitDecision::AverageFacing(u16 id, float facing) {
  u32 now = GetCurrentTick();
  auto it = facing_average.find(id);
  if (it == facing_average.end() || TICK_DIFF(now, it->second.tick) > (s32)(kFacingMemoryTicks * 5.0f)) {
    facing_average[id] = {facing, now};
    return facing;
  }
  float weight = 1.0f - expf(-(float)TICK_DIFF(now, it->second.tick) / kFacingMemoryTicks);
  it->second.value += (facing - it->second.value) * weight;
  it->second.tick = now;
  return it->second.value;
}

float CommitDecision::OwnBombDamageRate(const ShipSettings& ship, float bombs, float distance) {
  if (bombs <= 0.0f) return 0.0f;
  auto& arena = bot.game->connection.settings;
  return FireRate(ship.BombFireDelay) * kEngagedFireFraction * BombSelfDamageShare(distance) *
         (arena.BombDamageLevel / 1000.0f);
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

  // Other enemies near the bot that it can see add to the danger.
  auto& players = bot.game->player_manager;
  for (size_t i = 0; i < players.player_count; ++i) {
    Player& other = players.players[i];
    if (other.id == self.id || other.id == target.id || other.ship >= 8 || other.frequency == self.frequency) continue;
    if (other.position.Distance(self.position) > kThirdPartyDistance || !sight.Sees(other)) continue;
    incoming += DamageRate(other, self, false);
  }

  float target_spend = FiringCost(target_ship, GuessedLevel(target_ship.InitialGuns, target_ship.MaxGuns),
                                  GuessedLevel(target_ship.InitialBombs, target_ship.MaxBombs));
  float target_recharge = GuessedLevel(target_ship.InitialRecharge, target_ship.MaximumRecharge) / 10.0f;
  float outgoing = DamageRate(self, target, true);

  // A fire window: the target can't fire for a moment, which is worth the damage it would have dealt.
  self_energy += DamageRate(target, self, false) * opportunities.WindowSeconds(target);

  // Each side's own bombs hurt it at close range.
  float fight_distance = std::min(self.position.Distance(target.position), kFightDistance);
  float self_blast = OwnBombDamageRate(arena.ShipSettings[self.ship], (float)own.bombs, fight_distance);
  float target_blast = OwnBombDamageRate(
      target_ship, GuessedLevel(target_ship.InitialBombs, target_ship.MaxBombs), fight_distance);

  float self_loss = incoming + self_spend + self_blast - self_recharge;
  float target_loss = outgoing + target_spend + target_blast - target_recharge;

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
  auto grudge = memory ? memory->Grudge() : std::nullopt;
  if (grudge && *grudge == target.id) {
    commit -= kGrudgeAppetite;
    disengage -= kGrudgeAppetite;
  }
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

  if (mode == CombatMode::Commit) {
    if (self.position.Distance(target.position) > kCommitRange) {
      closing = true;
      if (chance >= disengage) return CombatMode::Commit;
    } else if (closing) {
      closing = false;
      if (chance >= commit - kHysteresis) return CombatMode::Commit;
    } else if (chance >= disengage) {
      return CombatMode::Commit;
    }
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

static bool IsBomb(const Weapon& weapon) {
  return (weapon.data.type == WeaponType::Bomb || weapon.data.type == WeaponType::ProximityBomb) &&
         !weapon.data.alternate;
}

void CommitDecision::HandleEvent(const WeaponSelfDamageEvent& event) {
  if (IsBomb(event.weapon)) bomb_damage_tick[event.shooter.id] = GetCurrentTick();
}

void CommitDecision::HandleEvent(const WeaponHitEvent& event) {
  Player* self = bot.game->player_manager.GetSelf();
  if (!notice || !self || !event.target || event.weapon.player_id != self->id || !IsBomb(event.weapon)) return;
  if (event.target->frequency == self->frequency) return;
  if (event.weapon.GetPosition().Distance(event.target->position) > kWonBlast) return;

  u32 now = GetCurrentTick();
  auto bombed = notice->LastBombSeen(event.target->id);
  if (!bombed || TICK_DIFF(now, *bombed) > (s32)kWonTicks) return;
  auto hurt = bomb_damage_tick.find(event.target->id);
  if (hurt != bomb_damage_tick.end() && !TICK_GT(*bombed, hurt->second)) return;  // their bomb landed

  next_decision_tick = now;
  if (bot.combat_logger) bot.combat_logger->WriteTactic("won exchange", event.target->name);
}

CombatMode CommitDecision::Strike(Player& self, Player& target) {
  u32 now = GetCurrentTick();
  if (mode != CombatMode::Commit) {
    commit_tick = now;
    closing = self.position.Distance(target.position) > kCommitRange;
  }
  mode = CombatMode::Commit;
  strike_until = now + kStrikeTicks;
  next_decision_tick = now + settings.decision_ticks;
  return mode;
}

CombatMode CommitDecision::Decide(Player& self, Player& target) {
  u32 now = GetCurrentTick();
  float distance = self.position.Distance(target.position);
  // A strike holds unless the bot's own energy runs low.
  if (mode == CombatMode::Commit && TICK_GT(strike_until, now) &&
      self.energy >= kStrikeAbortEnergy * bot.game->ship_controller.ship.energy) {
    return mode;
  }
  // Arriving from a closing-in: judge the fight afresh now.
  if (mode == CombatMode::Commit && closing && distance <= kCommitRange) next_decision_tick = now;
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
  if (settings.punish_energy && self.energy >= settings.commit_energy * bot.game->ship_controller.ship.energy &&
      target.ship < 8) {
    float target_max = (float)bot.game->connection.settings.ShipSettings[target.ship].MaximumEnergy;
    if (estimate.Get(target).energy < *settings.punish_energy * target_max) chosen = CombatMode::Commit;
  }
  if (chosen == CombatMode::Commit && mode != CombatMode::Commit) {
    commit_tick = now;
    closing = distance > kCommitRange;
  }
  mode = chosen;
  return mode;
}

}  // namespace chaos
}  // namespace zero
