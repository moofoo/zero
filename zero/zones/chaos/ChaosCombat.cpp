#include "ChaosCombat.h"

#include "CommitDecision.h"
#include "EnergyEstimate.h"
#include "Opportunities.h"
#include "PlayerHistory.h"
#include "TargetLock.h"
#include "WeaponNotice.h"

#include <zero/behavior/BehaviorBuilder.h>
#include <zero/behavior/BehaviorTree.h>
#include <zero/behavior/nodes/AimNode.h>
#include <zero/behavior/nodes/BlackboardNode.h>
#include <zero/behavior/nodes/InputActionNode.h>
#include <zero/behavior/nodes/MapNode.h>
#include <zero/behavior/nodes/MathNode.h>
#include <zero/behavior/nodes/MoveNode.h>
#include <zero/behavior/nodes/PlayerNode.h>
#include <zero/behavior/nodes/RenderNode.h>
#include <zero/behavior/nodes/ShipNode.h>
#include <zero/behavior/nodes/ThreatNode.h>
#include <zero/behavior/nodes/TimerNode.h>
#include <zero/behavior/nodes/WaypointNode.h>
#include <zero/CombatLogger.h>
#include <zero/RegionRegistry.h>
#include <zero/ZeroBot.h>
#include <zero/zones/svs/nodes/BurstAreaQueryNode.h>
#include <zero/zones/svs/nodes/DynamicPlayerBoundingBoxQueryNode.h>
#include <zero/zones/svs/nodes/FindNearestGreenNode.h>
#include <zero/zones/svs/nodes/IncomingDamageQueryNode.h>
#include <zero/zones/svs/nodes/MemoryTargetNode.h>

#include <math.h>

#include <random>
#include <string>

namespace zero {
namespace chaos {

using namespace svs;

// The closest a bot bombs an enemy without a BombMinRange (tiles): the SVS Warbird tree's.
constexpr float kDefaultBombMinRange = 10.0f;
// How loose the check that a shot's path crosses the target is: the box around the predicted intercept is the ship's
// radius times up to this (at full energy; down to 1 at none). The bullet box was 4 (zero's SVS Warbird); 6 measured 45%
// more bullets at the same hit rate on 2026-10-20 (chaos-offline docs/metrics/bomb-value-2026-10-20.md). The bomb box
// keeps the SVS Warbird's 3: a bomb's blast covers its misses.
constexpr float kBulletBoxScale = 6.0f;
constexpr float kBombBoxScale = 3.0f;
// How far around the target to count our own bullets heading at it, for bombing into them (tiles).
constexpr float kOutgoingDamageRadius = 40.0f;
// How far around the aim point shots are spread, and over what period.
constexpr float kShotSpread = 6.0f;
constexpr float kShotSpreadPeriod = 2.0f;

// Stands in for a trait the settings leave out: always fails, so the selector it's in moves on as if it weren't
// there.
struct FailureNode : public behavior::BehaviorNode {
  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override { return behavior::ExecuteResult::Failure; }
};

// Stands in for a check the settings leave out: always succeeds, so the sequence it's in carries on as if it weren't
// there. (zero's SuccessNode without a child fails.)
struct PassNode : public behavior::BehaviorNode {
  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override { return behavior::ExecuteResult::Success; }
};

// The commit decision's mode for this tick, from the blackboard (pressure before the first decision).
static CombatMode CurrentMode(behavior::ExecuteContext& ctx) {
  int mode = ctx.blackboard.ValueOr<int>("combat_mode", (int)CombatMode::Pressure);
  return mode >= 0 && mode < (int)CombatMode::Count ? (CombatMode)mode : CombatMode::Pressure;
}

// The share of max energy a mode keeps in reserve, below which no shot is worth firing. Backing off keeps enough for a
// few cover shots; recharging keeps the most.
static float ModeReserve(CombatMode mode) {
  switch (mode) {
    case CombatMode::Commit:
      return 0.1f;
    case CombatMode::Hold:
      return 0.35f;
    case CombatMode::Disengage:
      return 0.3f;
    case CombatMode::Recharge:
      return 0.5f;
    default:
      return 0.25f;
  }
}

// Writes a blackboard float multiplied by a constant to another key.
struct ScaleNode : public behavior::BehaviorNode {
  ScaleNode(const char* input_key, float scale, const char* output_key)
      : input_key(input_key), scale(scale), output_key(output_key) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    auto opt_value = ctx.blackboard.Value<float>(input_key);
    if (!opt_value) return behavior::ExecuteResult::Failure;

    ctx.blackboard.Set(output_key, *opt_value * scale);
    return behavior::ExecuteResult::Success;
  }

  const char* input_key;
  float scale;
  const char* output_key;
};

// Moves the aim point side to side around the target over time, so shots at range cover where they might dodge.
// From the SVS Terrier.
struct ShotSpreadNode : public behavior::BehaviorNode {
  ShotSpreadNode(const char* aimshot_key, float spread, float period)
      : aimshot_key(aimshot_key), spread(spread), period(period) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    if (!self || self->ship >= 8) return behavior::ExecuteResult::Failure;

    auto opt_aimshot = ctx.blackboard.Value<Vector2f>(aimshot_key);
    if (!opt_aimshot) return behavior::ExecuteResult::Failure;
    Vector2f aimshot = *opt_aimshot;

    Vector2f aim_direction = Normalize(aimshot - self->position);
    Vector2f perp = Perpendicular(aim_direction);

    float t = GetMicrosecondTick() / (kTickDurationMicro * 10.0f);
    aimshot += perp * sinf(t / period) * spread;

    ctx.blackboard.Set(aimshot_key, aimshot);

    return behavior::ExecuteResult::Success;
  }

  const char* aimshot_key = nullptr;
  float spread = 0.0f;
  float period = 1.0f;
};

// Writes the bot's estimate of a player's energy, and how unsure it is, to the blackboard, and logs it.
struct EnergyEstimateQueryNode : public behavior::BehaviorNode {
  EnergyEstimateQueryNode(EnergyEstimate* estimate, const char* player_key, const char* energy_key, const char* sd_key)
      : estimate(estimate), player_key(player_key), energy_key(energy_key), sd_key(sd_key) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    auto opt_player = ctx.blackboard.Value<Player*>(player_key);
    if (!opt_player || !*opt_player) return behavior::ExecuteResult::Failure;
    Player& player = **opt_player;

    EnergyEstimate::Reading reading = estimate->Get(player);
    ctx.blackboard.Set(energy_key, reading.energy);
    ctx.blackboard.Set(sd_key, reading.sd);

    if (ctx.bot->combat_logger) {
      float heuristic = ctx.bot->bot_controller->energy_tracker.GetEnergy(player);
      ctx.bot->combat_logger->WriteEstimate(player.name, reading.energy, reading.sd, heuristic);
    }

    return behavior::ExecuteResult::Success;
  }

  EnergyEstimate* estimate;
  const char* player_key;
  const char* energy_key;
  const char* sd_key;
};

// Runs the commit decision against the target, writes the mode to the blackboard as an int, and logs each change.
struct CommitDecisionNode : public behavior::BehaviorNode {
  CommitDecisionNode(CommitDecision* decision, const char* target_key) : decision(decision), target_key(target_key) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>(target_key);
    if (!self || !opt_target || !*opt_target) return behavior::ExecuteResult::Failure;

    CombatMode mode = decision->Decide(*self, **opt_target);
    if (mode != logged || !logged_once) {
      if (ctx.bot->combat_logger) ctx.bot->combat_logger->WriteMode(to_string(mode), decision->win_chance);
      logged = mode;
      logged_once = true;
    }
    ctx.blackboard.Set("combat_mode", (int)mode);
    return behavior::ExecuteResult::Success;
  }

  CommitDecision* decision;
  const char* target_key;
  CombatMode logged = CombatMode::Pressure;
  bool logged_once = false;
};

// Succeeds while the commit decision's mode is one of these.
struct CombatModeIsNode : public behavior::BehaviorNode {
  CombatModeIsNode(CombatMode mode, CombatMode or_mode) : mode(mode), or_mode(or_mode) {}
  explicit CombatModeIsNode(CombatMode mode) : mode(mode), or_mode(mode) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    int current = ctx.blackboard.ValueOr<int>("combat_mode", -1);
    bool match = current == (int)mode || current == (int)or_mode;
    return match ? behavior::ExecuteResult::Success : behavior::ExecuteResult::Failure;
  }

  CombatMode mode;
  CombatMode or_mode;
};

// Accepting a hit: while committing or pressuring, when incoming damage would make the bot dodge or repel, it judges
// the damage and its win chance after taking it, each with the error its Judgement allows. If it would still
// win (AcceptConfidence) and not drop below kLastEnergy, it sets "accepting": no dodge, repel or portal. Logs each new
// accept. Always succeeds.
struct AcceptNode : public behavior::BehaviorNode {
  AcceptNode(CommitDecision* decision, float accept_confidence, float dodge_threshold, float damage_judgement,
             float judgement_noise)
      : decision(decision),
        accept_confidence(accept_confidence),
        dodge_threshold(dodge_threshold),
        damage_judgement(damage_judgement),
        judgement_noise(judgement_noise),
        random(std::random_device{}()) {}

  // Never accept a hit expected to leave less than this share of max energy.
  static constexpr float kLastEnergy = 0.1f;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    bool was_accepting = ctx.blackboard.ValueOr<bool>("accepting", false);
    ctx.blackboard.Erase("accepting");

    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
    CombatMode mode = decision->Mode();
    if (!self || !opt_target || !*opt_target) return behavior::ExecuteResult::Success;
    if (mode != CombatMode::Commit && mode != CombatMode::Pressure) return behavior::ExecuteResult::Success;

    // Only damage that would otherwise make the bot dodge or repel needs a decision.
    float incoming = ctx.blackboard.ValueOr<float>("incoming_damage", 0.0f);
    float repel_energy = ctx.blackboard.ValueOr<float>("repel_energy", self->energy);
    float max_energy = (float)ctx.bot->game->ship_controller.ship.energy;
    if (incoming <= 0.0f || (incoming < repel_energy && incoming < dodge_threshold * max_energy)) {
      return behavior::ExecuteResult::Success;
    }

    float judged = incoming;
    if (damage_judgement > 0.0f) judged *= 1.0f + std::normal_distribution<float>(0.0f, damage_judgement)(random);
    if (judged < 0.0f) judged = 0.0f;
    if (self->energy - judged < kLastEnergy * max_energy) return behavior::ExecuteResult::Success;

    float chance = decision->WinChanceAfter(*self, **opt_target, judged);
    if (judgement_noise > 0.0f) chance += std::normal_distribution<float>(0.0f, judgement_noise)(random);
    if (chance < accept_confidence) return behavior::ExecuteResult::Success;

    ctx.blackboard.Set("accepting", true);
    if (!was_accepting && ctx.bot->combat_logger) ctx.bot->combat_logger->WriteAccept(judged, chance);
    return behavior::ExecuteResult::Success;
  }

  CommitDecision* decision;
  float accept_confidence;
  float dodge_threshold;
  float damage_judgement;
  float judgement_noise;
  std::mt19937 random;
};

// Follows the patrol route (the "waypoints" blackboard list, looping) under Newtonian movement. Ships thrust only along
// their heading, so pointing at a waypoint while drifting sideways can't remove the drift; these styles steer instead:
// - Correction: seek the next waypoint without facing it, so the ship turns to whatever correction its velocity needs.
// - Pursuit: as correction, towards a point `look_ahead` tiles further along the route, so turns start early and
//   waypoints are passed through rather than arrived at.
// - Cornering: as pursuit, braking before each corner to a speed that falls with how sharp it is (`corner_speed` at a
//   right angle), from the ship's thrust.
// Keeps "waypoint_index" (the waypoint being headed for) and "waypoint_position", and fails when the way ahead isn't
// clear, so the caller can path there instead.
enum class Navigation { Classic, Correction, Pursuit, Cornering };

struct RouteNode : public behavior::BehaviorNode {
  RouteNode(Navigation style, float look_ahead, float corner_speed)
      : style(style), look_ahead(look_ahead), corner_speed(corner_speed) {}

  // A waypoint counts as reached within this distance (tiles); classic switches at 15.
  static constexpr float kReachedDistance = 3.0f;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    auto opt_waypoints = ctx.blackboard.Value<std::vector<Vector2f>>("waypoints");
    if (!self || self->ship >= 8 || !opt_waypoints || opt_waypoints->empty()) return behavior::ExecuteResult::Failure;

    const std::vector<Vector2f>& route = *opt_waypoints;
    size_t count = route.size();
    size_t index = ctx.blackboard.ValueOr<size_t>("waypoint_index", 0) % count;

    // The leg from the previous waypoint to the one being headed for, and how far along it the ship is.
    Vector2f from = route[(index + count - 1) % count];
    Vector2f to = route[index];
    Vector2f leg = to - from;
    float leg_length = leg.Length();
    float along = leg_length > 0.0f ? (self->position - from).Dot(leg) / leg_length : 0.0f;

    // Past the end of the leg, or close to its waypoint: on to the next.
    float reached = style == Navigation::Correction ? 15.0f : kReachedDistance;
    if (self->position.Distance(to) <= reached || (style != Navigation::Correction && along >= leg_length)) {
      index = (index + 1) % count;
      from = to;
      to = route[index];
      leg = to - from;
      leg_length = leg.Length();
      along = leg_length > 0.0f ? (self->position - from).Dot(leg) / leg_length : 0.0f;
    }
    ctx.blackboard.Set("waypoint_index", index);
    ctx.blackboard.Set<Vector2f>("waypoint_position", to);

    // Where to steer: the waypoint, or a point look_ahead tiles along the route from the ship's place on it.
    Vector2f target = to;
    if (style == Navigation::Pursuit || style == Navigation::Cornering) {
      float remaining = look_ahead;
      float start = along < 0.0f ? 0.0f : along;
      size_t i = index;
      Vector2f a = from;
      Vector2f b = to;
      for (size_t step = 0; step < count; ++step) {
        float length = a.Distance(b);
        if (start + remaining <= length) {
          target = a + Normalize(b - a) * (start + remaining);
          break;
        }
        remaining -= length - start;
        start = 0.0f;
        i = (i + 1) % count;
        a = b;
        b = route[i];
        target = b;
      }
    }

    float radius = game.connection.settings.ShipSettings[self->ship].GetRadius();
    if (game.GetMap().CastShip(self, radius, target).hit) return behavior::ExecuteResult::Failure;

    float max_speed = Steering::GetMaxSpeed(game);
    float speed = max_speed;
    if (style == Navigation::Cornering) {
      // Brake to the corner's speed in time: v^2 = v_corner^2 + 2 * a * distance.
      Vector2f next_leg = route[(index + 1) % count] - to;
      float turn = acosf(fmaxf(-1.0f, fminf(1.0f, Normalize(leg).Dot(Normalize(next_leg)))));
      float sharpness = sinf(turn / 2.0f);  // 0 straight on, about 0.71 at a right angle, 1 turning back
      float corner = max_speed - (max_speed - corner_speed) * sharpness / 0.7071f;
      if (corner < corner_speed * 0.5f) corner = corner_speed * 0.5f;
      float acceleration = game.ship_controller.ship.thrust * (10.0f / 16.0f);
      float distance = self->position.Distance(to);
      float allowed = sqrtf(corner * corner + 2.0f * acceleration * distance);
      if (allowed < speed) speed = allowed;
    }

    auto& steering = ctx.bot->bot_controller->steering;
    Vector2f desired = Normalize(target - self->position) * speed;
    steering.force += desired - self->velocity;
    return behavior::ExecuteResult::Success;
  }

  Navigation style;
  float look_ahead;
  float corner_speed;
};

// Chooses the bot's target through its target lock, from the enemies it could fight: in a ship, not respawning, not
// in a safe zone, reachable over the map, and not vanished from where it was last seen (as NearestMemoryTargetNode).
struct LockedTargetNode : public behavior::BehaviorNode {
  LockedTargetNode(TargetLock* lock, const char* player_key) : lock(lock), player_key(player_key) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    if (!self) return behavior::ExecuteResult::Failure;

    candidates.clear();
    for (size_t i = 0; i < game.player_manager.player_count; ++i) {
      Player* player = game.player_manager.players + i;

      if (player->ship >= 8) continue;
      if (player->frequency == self->frequency) continue;
      if (player->IsRespawning()) continue;
      if (player->position == Vector2f(0, 0)) continue;
      // In view but sending nothing: they left where they were last seen.
      if (game.radar.InRadarView(player->position) && !game.player_manager.IsSynchronized(*player)) continue;
      if (!ctx.bot->bot_controller->region_registry->IsConnected(self->position, player->position)) continue;
      if (game.connection.map.GetTileId(player->position) == kTileIdSafe) continue;

      candidates.push_back(player);
    }

    Player* target = lock->Choose(candidates);
    if (lock->Battling()) {
      ctx.blackboard.Set("battle", true);
    } else {
      ctx.blackboard.Erase("battle");
    }
    if (!target) return behavior::ExecuteResult::Failure;

    ctx.blackboard.Set(player_key, target);
    return behavior::ExecuteResult::Success;
  }

  TargetLock* lock;
  const char* player_key;
  std::vector<Player*> candidates;
};

// How the target was moving as the bot saw it: `delay` ticks ago (a reaction time), led by `lead` of its velocity (1:
// a full intercept, 0: aiming where it was).
struct SkilledAimNode : public behavior::BehaviorNode {
  SkilledAimNode(PlayerHistory* history, const char* target_key, const char* position_key, u32 delay, float lead)
      : history(history), target_key(target_key), position_key(position_key), delay(delay), lead(lead) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>(target_key);
    if (!self || !opt_target || !*opt_target) return behavior::ExecuteResult::Failure;

    PlayerHistory::Sample seen = history->Get(**opt_target, delay);
    Vector2f aimshot = behavior::CalculateAimshot(game, *self, WeaponType::Bullet, seen.position, seen.velocity * lead);
    ctx.blackboard.Set(position_key, aimshot);
    return behavior::ExecuteResult::Success;
  }

  PlayerHistory* history;
  const char* target_key;
  const char* position_key;
  u32 delay;
  float lead;
};

// The long-range opening: on a new target far away, with plenty of energy and no recent hits, the bot flies straight
// at it, bombing as the shot choice allows, to weaken it before the fight proper ("fly at him at full speed and let off
// two or three bombs"). Taken with the bot's spotting chance and logged as an "opening" tactic. It ends at the leash
// distance, when incoming damage would make the bot dodge, below kEndEnergy, or after kMaxTicks; then the commit
// decision's modes take over. Checked every tick, before movement, so a dodge or a target change isn't missed; sets
// "opening" while it lasts. Always succeeds.
struct OpeningCheckNode : public behavior::BehaviorNode {
  OpeningCheckNode(Opportunities* opportunities, std::optional<float> dodge_threshold)
      : opportunities(opportunities), dodge_threshold(dodge_threshold) {}

  static constexpr float kMinDistance = 40.0f;
  static constexpr float kStartEnergy = 0.8f;
  static constexpr float kEndEnergy = 0.6f;
  static constexpr float kQuietSeconds = 3.0f;
  static constexpr u32 kMaxTicks = 600;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    ctx.blackboard.Erase("opening");
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
    auto opt_leash = ctx.blackboard.Value<float>("leash_distance");
    if (!self || !opt_target || !*opt_target || !opt_leash) return behavior::ExecuteResult::Success;
    Player& target = **opt_target;
    float max_energy = (float)ctx.bot->game->ship_controller.ship.energy;
    float distance = self->position.Distance(target.position);
    u32 now = GetCurrentTick();

    if (target.id != target_id) {
      target_id = target.id;
      auto hit = opportunities->LastHitTick();
      bool quiet = !hit || TICK_DIFF(now, *hit) >= (s32)(kQuietSeconds * 100.0f);
      active = distance >= kMinDistance && self->energy >= kStartEnergy * max_energy && quiet && opportunities->Spot();
      end_tick = now + kMaxTicks;
      if (active && ctx.bot->combat_logger) ctx.bot->combat_logger->WriteTactic("opening", target.name);
    }
    if (!active) return behavior::ExecuteResult::Success;

    float incoming = ctx.blackboard.ValueOr<float>("incoming_damage", 0.0f);
    bool would_dodge = dodge_threshold && incoming >= *dodge_threshold * max_energy;
    if (distance <= *opt_leash || self->energy < kEndEnergy * max_energy || TICK_GT(now, end_tick) || would_dodge) {
      active = false;
      return behavior::ExecuteResult::Success;
    }
    ctx.blackboard.Set("opening", true);
    return behavior::ExecuteResult::Success;
  }

  Opportunities* opportunities;
  std::optional<float> dodge_threshold;
  u16 target_id = 0xFFFF;
  bool active = false;
  u32 end_tick = 0;
};

// Thrust timing in a fight (ticket 19): the ship thrusts only along its heading and fires the way it faces, so while
// aiming, thrust helps only when the heading happens to line up with the move the bot wants. Sets the actuator's thrust
// threshold for this tick, the mode's alignment times the bot's Piloting: committing keeps the aim and thrusts only when
// well aligned; backing off thrusts more freely, backwards too, still facing the enemy. Always succeeds.
struct ThrustTimingNode : public behavior::BehaviorNode {
  explicit ThrustTimingNode(float piloting) : piloting(piloting) {}

  // The alignment (cosine) a mode needs to thrust.
  static float Alignment(CombatMode mode) {
    switch (mode) {
      case CombatMode::Commit:
        return 0.7f;
      case CombatMode::Disengage:
      case CombatMode::Recharge:
        return 0.3f;
      default:
        return 0.5f;
    }
  }

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    ctx.bot->bot_controller->steering.thrust_threshold = Alignment(CurrentMode(ctx)) * piloting;
    return behavior::ExecuteResult::Success;
  }

  float piloting;
};

// Aims where the target will be after bouncing off a wall, when its path meets one before the shot can reach it: the
// guide's rebound. The target's flight, as the bot saw it `delay` ticks ago, is simulated with the game's own movement
// and bounces; the aim is left alone when there's no bounce first. For bots that lead their shots fully. Bullets and
// bombs travel at almost the same speed in Chaos, so the bullet's serves both, as for the aim itself. Logs a "rebound"
// tactic at most once a second.
struct ReboundAimNode : public behavior::BehaviorNode {
  ReboundAimNode(PlayerHistory* history, u32 delay, const char* target_key, const char* position_key)
      : history(history), delay(delay), target_key(target_key), position_key(position_key) {}

  static constexpr float kStep = 0.02f;       // seconds
  static constexpr float kMaxSeconds = 3.0f;  // no further ahead than this
  static constexpr float kFarDistance = 50.0f;
  static constexpr u32 kLogTicks = 100;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>(target_key);
    if (!self || self->ship >= 8 || !opt_target || !*opt_target) return behavior::ExecuteResult::Success;
    Player& target = **opt_target;
    float weapon_speed = behavior::GetWeaponSpeed(game, *self, WeaponType::Bullet);
    if (weapon_speed <= 0.0f) return behavior::ExecuteResult::Success;

    // A copy of the target, as last seen, flown forward until a shot fired now could meet it.
    PlayerHistory::Sample seen = history->Get(target, delay);
    Player ghost = target;
    ghost.position = seen.position;
    ghost.velocity = seen.velocity;
    ghost.lerp_time = 0.0f;
    bool bounced = false;
    for (float t = kStep; t <= kMaxSeconds; t += kStep) {
      Vector2f before = ghost.velocity;
      game.player_manager.SimulatePlayer(ghost, kStep, true);
      if (ghost.velocity.x * before.x < 0.0f || ghost.velocity.y * before.y < 0.0f) bounced = true;

      Vector2f offset = ghost.position - self->position - self->velocity * t;
      if (offset.Length() > weapon_speed * t) continue;
      if (!bounced || ghost.position.Distance(seen.position) > kFarDistance) break;

      ctx.blackboard.Set(position_key, self->position + offset);
      u32 now = GetCurrentTick();
      if (ctx.bot->combat_logger && !TICK_GT(next_log_tick, now)) {
        ctx.bot->combat_logger->WriteTactic("rebound", target.name);
        next_log_tick = now + kLogTicks;
      }
      break;
    }
    return behavior::ExecuteResult::Success;
  }

  PlayerHistory* history;
  u32 delay;
  const char* target_key;
  const char* position_key;
  u32 next_log_tick = 0;
};

// Dodging incoming damage, as a player commits to a dodge: when the weapons the bot has noticed (WeaponNotice) would
// do at least the threshold fraction of its max energy, or kill it, it picks a side away from their path and pushes
// that way for at least kDodgeHoldTicks, or until nothing has been coming for kClearTicks, and succeeds. With
// DodgeError's chance the side is the wrong one, decided when the dodge starts. Below the threshold it gives a small
// push away and fails, letting the rest of the tree run.
struct CommittedDodgeNode : public behavior::BehaviorNode {
  CommittedDodgeNode(float threshold, float error, WeaponNotice* notice)
      : threshold(threshold),
        error(error),
        notice(notice),
        incoming(threshold, kDodgeDistance),
        random(std::random_device{}()) {}

  static constexpr float kDodgeDistance = 35.0f;
  static constexpr u32 kDodgeHoldTicks = 40;
  // Weapons flicker in and out of the check (a mine as the bot passes it), so a dodge ends only once nothing has been
  // coming for this long.
  static constexpr u32 kClearTicks = 10;
  static constexpr float kStrongForce = 10000.0f;
  static constexpr float kMinimumForce = 2.0f;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    if (!self || self->ship >= 8) return behavior::ExecuteResult::Failure;

    behavior::IncomingDamageReport report = incoming.GetIncomingDamage(ctx, self, kDodgeDistance);
    u32 now = GetCurrentTick();
    auto& steering = ctx.bot->bot_controller->steering;

    if (report.weapon_count == 0) {
      if (active && TICK_DIFF(now, last_incoming_tick) < (s32)kClearTicks &&
          TICK_DIFF(now, start_tick) < (s32)kDodgeHoldTicks) {
        steering.force = held_side * kStrongForce;
        return behavior::ExecuteResult::Success;
      }
      active = false;
      return behavior::ExecuteResult::Failure;
    }
    last_incoming_tick = now;

    float damage = report.weapon_count * report.average_damage;
    float damage_percent = damage / (float)ctx.bot->game->ship_controller.ship.energy;
    bool strong = damage_percent >= threshold || self->energy - damage <= 0;

    Ray ray(report.average_origin, Normalize(report.average_direction));
    Vector2f side = Normalize(self->position - ray.GetClosestPosition(self->position));

    if (strong && !active) {
      active = true;
      start_tick = now;
      wrong = error > 0.0f && std::uniform_real_distribution<float>(0.0f, 1.0f)(random) < error;
      held_side = wrong ? side * -1.0f : side;
      LogDodge(ctx, report, wrong);
    }

    if (active && (strong || TICK_DIFF(now, start_tick) < kDodgeHoldTicks)) {
      steering.force = held_side * kStrongForce;
      return behavior::ExecuteResult::Success;
    }

    active = false;
    steering.force = side * (kMinimumForce + damage_percent * 10.0f);
    return behavior::ExecuteResult::Failure;
  }

  void LogDodge(behavior::ExecuteContext& ctx, const behavior::IncomingDamageReport& report, bool wrong_side) {
    if (!ctx.bot->combat_logger || !report.main_weapon) return;
    const Weapon& weapon = *report.main_weapon;
    WeaponManager& weapons = ctx.bot->game->weapon_manager;
    u32 now = GetCurrentTick();
    s32 alive = weapons.GetWeaponTotalAliveTime(weapon.data.type, weapon.data.alternate);
    int age = alive - (int)TICK_DIFF(weapon.end_tick, now);
    int noticed = -1;
    if (notice) {
      auto at = notice->NoticedAt(weapon);
      if (at) noticed = (int)TICK_DIFF(now, *at);
    }
    ctx.bot->combat_logger->WriteDodge(weapon.data, report.main_distance, age, noticed, wrong_side);
  }

  float threshold;
  float error;
  WeaponNotice* notice;
  behavior::DodgeIncomingDamage incoming;
  bool active = false;
  bool wrong = false;
  u32 start_tick = 0;
  u32 last_incoming_tick = 0;
  Vector2f held_side;
  std::mt19937 random;
};

// Turns the aim point around the bot by an angle that drifts smoothly (mean-reverting, like a hand), about
// `degrees` off on average.
struct AimErrorNode : public behavior::BehaviorNode {
  AimErrorNode(const char* position_key, float degrees)
      : position_key(position_key), sd(Radians(degrees)), random(std::random_device{}()) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_position = ctx.blackboard.Value<Vector2f>(position_key);
    if (!self || !opt_position) return behavior::ExecuteResult::Failure;

    // An Ornstein-Uhlenbeck step over the ticks since the last one.
    u32 now = GetCurrentTick();
    s32 ticks = last_tick ? TICK_DIFF(now, last_tick) : kDriftTicks;
    last_tick = now;
    if (ticks > 0) {
      float keep = expf(-ticks / kDriftTicks);
      angle = angle * keep + sd * sqrtf(1.0f - keep * keep) * std::normal_distribution<float>(0.0f, 1.0f)(random);
    }

    Vector2f offset = *opt_position - self->position;
    float c = cosf(angle), s = sinf(angle);
    Vector2f turned(offset.x * c - offset.y * s, offset.x * s + offset.y * c);
    ctx.blackboard.Set(position_key, self->position + turned);

    return behavior::ExecuteResult::Success;
  }

  // How long the drift takes to wander, in ticks: about half a second.
  static constexpr float kDriftTicks = 50.0f;

  const char* position_key;
  float sd;
  float angle = 0.0f;
  u32 last_tick = 0;
  std::mt19937 random;
};

// Whether the bot, backing off (disengage or recharge), is being chased by its target: within this distance (tiles)
// and closing at this speed or more (tiles/s), as `chaos.py stats` judges a pursuit. Sets "pursued", and while it
// holds, "pursuer_distance". Always succeeds.
struct PursuitNode : public behavior::BehaviorNode {
  static constexpr float kPursuitDistance = 30.0f;
  static constexpr float kPursuitSpeed = 2.0f;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    ctx.blackboard.Erase("pursued");
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
    CombatMode mode = CurrentMode(ctx);
    bool backing_off = mode == CombatMode::Disengage || mode == CombatMode::Recharge;
    if (!self || !opt_target || !*opt_target || !backing_off) return behavior::ExecuteResult::Success;

    Player& pursuer = **opt_target;
    float distance = pursuer.position.Distance(self->position);
    Vector2f toward_self = Normalize(self->position - pursuer.position);
    float closing = (pursuer.velocity - self->velocity).Dot(toward_self);
    if (distance <= kPursuitDistance && closing >= kPursuitSpeed) {
      ctx.blackboard.Set("pursued", true);
      ctx.blackboard.Set("pursuer_distance", distance);
    }
    return behavior::ExecuteResult::Success;
  }
};

// A cover shot while pursued: fired back at the pursuer however slowly it travels, to make following in a straight
// line risky, while the pursuer is roughly in front within range and the bot keeps its reserve.
struct CoverShotNode : public behavior::BehaviorNode {
  static constexpr float kCoverRange = 25.0f;
  static constexpr float kFacingCos = 0.7f;  // within about 45 degrees
  static constexpr float kReserve = 0.3f;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
    if (!self || !opt_target || !*opt_target || !ctx.blackboard.ValueOr<bool>("pursued", false)) {
      return behavior::ExecuteResult::Failure;
    }

    Vector2f to_pursuer = (*opt_target)->position - self->position;
    if (to_pursuer.Length() > kCoverRange) return behavior::ExecuteResult::Failure;
    if (self->GetHeading().Dot(Normalize(to_pursuer)) < kFacingCos) return behavior::ExecuteResult::Failure;
    if (self->energy < kReserve * ctx.bot->game->ship_controller.ship.energy) return behavior::ExecuteResult::Failure;
    return behavior::ExecuteResult::Success;
  }
};

// Lays a mine where the bot is while pursued, for the pursuer to fly over: when the pursuer is between kMinRange and
// `range` tiles away, at most every kIntervalTicks, keeping the reserve after the mine's cost, within the arena's mine
// cap. With `on_trail`, only when the pursuer is heading at the bot (so it will cross the spot).
struct MineCoverNode : public behavior::BehaviorNode {
  MineCoverNode(float range, bool on_trail) : range(range), on_trail(on_trail) {}

  static constexpr float kMinRange = 8.0f;
  static constexpr u32 kIntervalTicks = 200;
  static constexpr float kReserve = 0.3f;
  static constexpr float kTrailCos = 0.87f;  // within about 30 degrees

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
    if (!self || self->ship >= 8 || !opt_target || !*opt_target) return behavior::ExecuteResult::Failure;
    if (!ctx.blackboard.ValueOr<bool>("pursued", false)) return behavior::ExecuteResult::Failure;

    u32 now = GetCurrentTick();
    if (TICK_GT(next_mine_tick, now)) return behavior::ExecuteResult::Failure;

    float distance = ctx.blackboard.ValueOr<float>("pursuer_distance", 0.0f);
    if (distance < kMinRange || distance > range) return behavior::ExecuteResult::Failure;

    Player& pursuer = **opt_target;
    if (on_trail && pursuer.velocity.LengthSq() > 0.0f &&
        Normalize(pursuer.velocity).Dot(Normalize(self->position - pursuer.position)) < kTrailCos) {
      return behavior::ExecuteResult::Failure;
    }

    Ship& ship = game.ship_controller.ship;
    auto& settings = game.connection.settings.ShipSettings[self->ship];
    if (ship.bombs == 0) return behavior::ExecuteResult::Failure;
    float cost = settings.LandmineFireEnergy + settings.LandmineFireEnergyUpgrade * (ship.bombs - 1.0f);
    if (self->energy - cost < kReserve * ship.energy) return behavior::ExecuteResult::Failure;

    // The arena's cap on mines out at once.
    u32 mines = 0;
    for (size_t i = 0; i < game.weapon_manager.weapon_count; ++i) {
      Weapon& weapon = game.weapon_manager.weapons[i];
      bool mine = (weapon.data.type == WeaponType::Bomb || weapon.data.type == WeaponType::ProximityBomb) &&
                  weapon.data.alternate;
      if (mine && weapon.player_id == self->id) ++mines;
    }
    if (mines >= settings.MaxMines) return behavior::ExecuteResult::Failure;

    auto& input = ctx.bot->bot_controller->input;
    if (!input) return behavior::ExecuteResult::Failure;
    input->SetAction(InputAction::Mine, true);
    input->OnAction(InputAction::Mine);
    next_mine_tick = now + kIntervalTicks;
    return behavior::ExecuteResult::Success;
  }

  float range;
  bool on_trail;
  u32 next_mine_tick = 0;
};

// Shot choice by expected value, with the commit decision: fire when the shot's chance of hitting times its damage is
// worth the energy it costs. Energy is cheap with plenty to spare and dear near the reserve the bot's mode needs, and
// below the reserve no shot is worth it. A shot slower than the weapon's own speed towards the target is less likely
// to land. A bot that judges shots poorly overestimates their chance, up to double, and a bomb's worth is scaled by the
// bot's appetite for bombing. A slow target the bot has spotted is likelier to be hit. A bomb spends a lot of energy at
// once and blocks bullets after it, so it must also bring at least kMinBombEfficiency damage per energy, as the bot
// judges it: long, low-odds bombs are left for bullets (ticket 22). A bot with poor judgement overrates its bombs, so
// it bombs from further than it should, a novice's mistake.
struct ShotValueNode : public behavior::BehaviorNode {
  ShotValueNode(WeaponType weapon, const char* velocity_key, float judgement, float bomb_appetite, float lead_skill,
                Opportunities* opportunities)
      : weapon(weapon),
        velocity_key(velocity_key),
        judgement(judgement),
        bomb_appetite(bomb_appetite),
        lead_skill(lead_skill),
        opportunities(opportunities) {}

  // A bomb's least damage per energy: with the Chaos Warbird's 350-energy bomb, about a veteran's bomb at 50 tiles or a
  // novice's at 40.
  static constexpr float kMinBombEfficiency = 0.35f;
  // Bullets keep only this share of the mode's reserve: a player keeps firing bullets as their energy falls, and a
  // bullet spends little at a time (ticket 22).
  static constexpr float kBulletReserveShare = 0.5f;

  // With this much more than the reserve, energy is at its cheapest; at the reserve, its dearest.
  static constexpr float kPlenty = 0.3f;
  static constexpr float kCheapest = 0.1f;
  static constexpr float kDearest = 2.0f;

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    auto opt_velocity = ctx.blackboard.Value<Vector2f>(velocity_key);
    auto opt_target = ctx.blackboard.Value<Vector2f>("nearest_target_position");
    auto opt_player = ctx.blackboard.Value<Player*>("nearest_target");
    if (!self || self->ship >= 8 || !opt_velocity || !opt_target) return behavior::ExecuteResult::Failure;

    auto& arena = game.connection.settings;
    auto& ship_settings = arena.ShipSettings[self->ship];
    Ship& ship = game.ship_controller.ship;
    bool bomb = weapon == WeaponType::Bomb;

    // How fast the shot heads at the target compared with the weapon's own speed.
    Vector2f to_target = *opt_target - self->position;
    float distance = to_target.Length();
    float weapon_speed = (bomb ? ship_settings.BombSpeed : ship_settings.BulletSpeed) / 16.0f / 10.0f;
    float speed = opt_velocity->Dot(Normalize(to_target));
    if (speed <= 0.0f || weapon_speed <= 0.0f) return behavior::ExecuteResult::Failure;
    float speed_factor = speed / weapon_speed < 1.0f ? speed / weapon_speed : 1.0f;
    float slow_factor = opt_player && *opt_player ? opportunities->SlowFactor(**opt_player, bomb, distance) : 1.0f;

    // The shot's expected damage, as the bot judges it, and its energy cost.
    float value;
    float cost;
    if (bomb) {
      float level = ship.bombs > 0 ? ship.bombs - 1.0f : 0.0f;
      value = BombDamageShare(distance, lead_skill) * (arena.BombDamageLevel / 1000.0f) * speed_factor * slow_factor *
              bomb_appetite;
      cost = ship_settings.BombFireEnergy + ship_settings.BombFireEnergyUpgrade * level;
    } else {
      float guns = ship.guns > 0 ? (float)ship.guns : 1.0f;
      float chance = BulletHitChance(distance) * speed_factor * slow_factor;
      value = (chance < 1.0f ? chance : 1.0f) * BulletDamage(arena, guns);
      cost = ship_settings.BulletFireEnergy * guns;
    }
    value *= 2.0f - judgement;  // poor judgement overestimates

    // What energy is worth now, from how much the bot has over its mode's reserve.
    float reserve = ModeReserve(CurrentMode(ctx)) * (bomb ? 1.0f : kBulletReserveShare);
    float spare = self->energy / (float)ship.energy - reserve;
    if (spare <= 0.0f) return behavior::ExecuteResult::Failure;
    float worth = spare >= kPlenty ? kCheapest : kDearest + (kCheapest - kDearest) * (spare / kPlenty);

    // Energy's price for this shot: a bomb's never below its least efficiency.
    float price = bomb && worth < kMinBombEfficiency ? kMinBombEfficiency : worth;
    return value >= cost * price ? behavior::ExecuteResult::Success : behavior::ExecuteResult::Failure;
  }

  WeaponType weapon;
  const char* velocity_key;
  float judgement;
  float bomb_appetite;
  float lead_skill;
  Opportunities* opportunities;
};

// Steer by thrust, aim to shoot (ticket 27). A player in a fight doesn't keep their nose on the enemy: they face where
// they want to go and thrust hard, swing onto the target to shoot, and turn away again. After the movement has set its
// steering force (and facing), this alternates two phases:
// - Steer: face the wanted move or its exact opposite, whichever is the smaller turn, and thrust forward or in reverse
//   once nearly aligned. The move is an orbit: circle the target at a cruising speed that varies, keeping the
//   movement's push towards or away from it (so a commit spirals in and backing off spirals out), now and then
//   switching direction. Circling never lets the needed change of velocity fade, so the bot keeps thrusting and turning
//   at speed, as a player does, and is hard to lead. Dodges steer without the orbit, thrusting sooner in the turn.
// - Aim: keep facing the aim point (thrust timed as ticket 19 has it), from when a shot is possible (bullets in range,
//   or a bomb beyond the bot's minimum bombing range, with the weapon ready and energy above the mode's reserve) until
//   a bomb or a burst of bullets is fired, or kMaxAimTicks pass.
// Steering lasts at least kMinSteerTicks before the bot can aim again. Each steering phase is taken with the bot's
// Piloting as its chance; one not taken means another aim phase, as a bot that doesn't pilot always aims. Shots are
// counted from the bot's own fire events. Always succeeds.
struct FlightPhaseNode : public behavior::BehaviorNode, EventHandler<WeaponFireEvent> {
  FlightPhaseNode(float piloting, float bomb_min_range, WeaponNotice* notice)
      : piloting(piloting), bomb_min_range(bomb_min_range), notice(notice), random(std::random_device{}()) {}

  static constexpr float kShotRange = 40.0f;        // bullets are worth aiming for within this (tiles)
  static constexpr u32 kMaxAimTicks = 100;          // an aim phase lasts at most this long
  static constexpr u32 kMinSteerTicks = 25;         // a steer phase lasts at least this long
  static constexpr int kBurstBullets = 3;           // an aim phase ends after this many bullets
  static constexpr float kAlignedCos = 0.7f;        // steering thrusts once within about 45 degrees, either way
  static constexpr float kDodgeAlignedCos = 0.3f;   // a dodge thrusts once within about 72 degrees
  static constexpr float kSwitchCos = 0.2f;         // forward and reverse swap only past this margin
  static constexpr float kCruiseMinSpeed = 13.0f;   // orbiting speed, picked afresh every 1-2 seconds (tiles/s)
  static constexpr float kCruiseMaxSpeed = 19.0f;
  static constexpr u32 kCruiseMinTicks = 100;
  static constexpr u32 kCruiseMaxTicks = 200;
  static constexpr u32 kSwitchMinTicks = 300;       // the orbit switches direction every 3-6 seconds
  static constexpr u32 kSwitchMaxTicks = 600;
  static constexpr u32 kGapTicks = 20;              // not run for this long: start afresh, steering
  static constexpr float kFacePointDistance = 10.0f;  // how far ahead the point to face is put (tiles)
  // Sideways early: an enemy on screen within this many tiles, its nose within about 20 degrees of the bot, that could
  // have a bomb ready (it hasn't been seen bombing within its bomb delay), has the bot steer across its line of fire:
  // with full Piloting, at least the top orbiting speed and with this much of the speed along the line taken off.
  static constexpr float kSidestepRange = 30.0f;
  static constexpr float kSidestepCos = 0.94f;
  static constexpr float kSidestepAlongCut = 0.8f;

  void HandleEvent(const WeaponFireEvent& event) override {
    if (event.player.id != self_id) return;
    WeaponType type = event.data.type;
    if (type == WeaponType::Bullet || type == WeaponType::BouncingBullet) {
      ++bullets;
    } else if ((type == WeaponType::Bomb || type == WeaponType::ProximityBomb) && !event.data.alternate) {
      bomb_fired = true;
    }
  }

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Game& game = *ctx.bot->game;
    Player* self = game.player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
    auto opt_aimshot = ctx.blackboard.Value<Vector2f>("aimshot");
    if (!self || self->ship >= 8 || !opt_target || !*opt_target || !opt_aimshot) return behavior::ExecuteResult::Success;
    Player& target = **opt_target;
    self_id = self->id;
    u32 now = GetCurrentTick();
    if (TICK_DIFF(now, last_run_tick) > (s32)kGapTicks) StartSteer(now);
    last_run_tick = now;
    bool dodging = ctx.blackboard.Has("dodging");

    if (aiming) {
      bool done = bomb_fired || bullets >= kBurstBullets || TICK_DIFF(now, phase_start) >= (s32)kMaxAimTicks;
      if (dodging) {
        StartSteer(now);
      } else if (done) {
        if (Roll()) {
          StartSteer(now);
        } else {
          StartAim(now);
        }
      }
    } else if (!dodging && TICK_DIFF(now, phase_start) >= (s32)kMinSteerTicks && ShotPossible(ctx, *self, target, now)) {
      StartAim(now);
    }

    auto& steering = ctx.bot->bot_controller->steering;
    if (aiming) {
      steering.rotation = 0.0f;
      steering.Face(game, *opt_aimshot);
      return behavior::ExecuteResult::Success;
    }

    // Steering: the orbit's velocity, with the movement's push towards or away from the target, less the velocity the
    // bot has: the change it needs, which circling keeps from fading.
    Vector2f move = steering.force;
    Vector2f heading_for = move;
    Vector2f to_target = target.position - self->position;
    if (!dodging && to_target.LengthSq() > 0.0f) {
      if (!TICK_GT(next_switch_tick, now)) {
        side = -side;
        next_switch_tick = now + std::uniform_int_distribution<u32>(kSwitchMinTicks, kSwitchMaxTicks)(random);
      }
      if (!TICK_GT(next_cruise_tick, now)) {
        cruise_speed = std::uniform_real_distribution<float>(kCruiseMinSpeed, kCruiseMaxSpeed)(random);
        next_cruise_tick = now + std::uniform_int_distribution<u32>(kCruiseMinTicks, kCruiseMaxTicks)(random);
      }
      Vector2f toward = Normalize(to_target);
      Vector2f across = Perpendicular(toward) * side;
      // The movement's force is its wanted velocity less the bot's; its wanted velocity towards the target is kept.
      float radial = (move + self->velocity).Dot(toward);
      Vector2f wanted = toward * radial + across * cruise_speed;
      Player* threat = BombThreat(ctx, *self, now);
      if (threat) {
        // Across the threat's line of fire, keeping the way the bot is already going across it.
        Vector2f line = Normalize(self->position - threat->position);
        Vector2f lateral = Perpendicular(line);
        float across_speed = wanted.Dot(lateral);
        float sign = across_speed > 0.0f ? 1.0f : across_speed < 0.0f ? -1.0f : side;
        float speed = fabsf(across_speed);
        speed += piloting * std::max(0.0f, kCruiseMaxSpeed - speed);
        float along = wanted.Dot(line) * (1.0f - piloting * kSidestepAlongCut);
        wanted = line * along + lateral * (sign * speed);
        if (threat->id != last_threat_id && ctx.bot->combat_logger) {
          ctx.bot->combat_logger->WriteTactic("sidestep", threat->name);
        }
      }
      last_threat_id = threat ? threat->id : 0xFFFF;
      heading_for = wanted - self->velocity;
      steering.force = heading_for;
    }
    if (heading_for.LengthSq() <= 0.0f) return behavior::ExecuteResult::Success;

    // Forward or reverse, whichever is the smaller turn, keeping the last choice near the middle.
    Vector2f direction = Normalize(heading_for);
    float alignment = self->GetHeading().Dot(direction);
    if (alignment > kSwitchCos) {
      reverse = false;
    } else if (alignment < -kSwitchCos) {
      reverse = true;
    }
    Vector2f facing = reverse ? direction * -1.0f : direction;
    steering.rotation = 0.0f;
    steering.Face(game, self->position + facing * kFacePointDistance);
    steering.thrust_threshold = dodging ? kDodgeAlignedCos : kAlignedCos;
    return behavior::ExecuteResult::Success;
  }

  bool ShotPossible(behavior::ExecuteContext& ctx, const Player& self, const Player& target, u32 now) {
    Ship& ship = ctx.bot->game->ship_controller.ship;
    if (self.energy <= ModeReserve(CurrentMode(ctx)) * ship.energy) return false;
    float distance = self.position.Distance(target.position);
    bool bullet = distance <= kShotRange && !TICK_GT(ship.next_bullet_tick, now);
    bool bomb = ship.bombs > 0 && distance >= bomb_min_range && !TICK_GT(ship.next_bomb_tick, now);
    return bullet || bomb;
  }

  // The nearest enemy that could bomb the bot now: on its screen, within kSidestepRange, nose on it, and not seen
  // bombing within its ship's bomb delay (a bot that doesn't keep track of noticed weapons assumes a bomb is ready).
  Player* BombThreat(behavior::ExecuteContext& ctx, const Player& self, u32 now) {
    Game& game = *ctx.bot->game;
    Vector2f extent = ctx.blackboard.ValueOr<Vector2f>("view_extent", Vector2f(1e6f, 1e6f));
    Player* nearest = nullptr;
    float nearest_distance = kSidestepRange;
    for (size_t i = 0; i < game.player_manager.player_count; ++i) {
      Player& enemy = game.player_manager.players[i];
      if (enemy.id == self.id || enemy.ship >= 8 || enemy.frequency == self.frequency || enemy.enter_delay > 0.0f) {
        continue;
      }
      Vector2f delta = self.position - enemy.position;
      if (fabsf(delta.x) > extent.x || fabsf(delta.y) > extent.y) continue;
      float distance = delta.Length();
      if (distance <= 0.0f || distance > nearest_distance) continue;
      if (enemy.GetHeading().Dot(delta * (1.0f / distance)) < kSidestepCos) continue;
      if (notice) {
        auto last = notice->LastBombSeen(enemy.id);
        u32 delay = game.connection.settings.ShipSettings[enemy.ship].BombFireDelay;
        if (last && TICK_DIFF(now, *last) < (s32)delay) continue;
      }
      nearest = &enemy;
      nearest_distance = distance;
    }
    return nearest;
  }

  bool Roll() {
    return piloting >= 1.0f || std::uniform_real_distribution<float>(0.0f, 1.0f)(random) < piloting;
  }

  void StartAim(u32 now) {
    aiming = true;
    phase_start = now;
    bullets = 0;
    bomb_fired = false;
  }

  void StartSteer(u32 now) {
    aiming = false;
    phase_start = now;
  }

  u16 self_id = 0xFFFF;  // the bot's player, for its own fire events
  float piloting;
  float bomb_min_range;
  WeaponNotice* notice;
  u16 last_threat_id = 0xFFFF;
  std::mt19937 random;
  bool aiming = false;
  u32 phase_start = 0;
  u32 last_run_tick = 0;
  int bullets = 0;
  bool bomb_fired = false;
  bool reverse = false;
  float side = 1.0f;
  u32 next_switch_tick = 0;
  float cruise_speed = kCruiseMinSpeed;
  u32 next_cruise_tick = 0;
};

// Steering and aiming in phases, for bots that pilot at all. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateFlightPhases(const CombatSettings& settings,
                                                                  WeaponNotice* notice) {
  if (settings.piloting <= 0.0f) return std::make_unique<PassNode>();

  return std::make_unique<FlightPhaseNode>(settings.piloting, settings.bomb_min_range.value_or(kDefaultBombMinRange),
                                           notice);
}

// Dodge when incoming damage is at least the threshold fraction of max energy, unless the bot has decided to accept
// the hit. The dodge is a push sideways from the incoming weapons, held to one side, sometimes the wrong one
// (CommittedDodgeNode). A bot that doesn't pilot keeps facing its aim point while it dodges, so aiming and firing carry
// on; one that does turns to the dodge (FlightPhaseNode), which the "dodging" flag set here tells it.
static std::unique_ptr<behavior::BehaviorNode> CreateDodge(const CombatSettings& settings, bool keep_facing,
                                                           WeaponNotice* notice) {
  using namespace behavior;
  if (!settings.dodge_threshold) return std::make_unique<FailureNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .InvertChild<BlackboardSetQueryNode>("accepting")
        .Child<CommittedDodgeNode>(*settings.dodge_threshold, settings.dodge_error, notice)
        .Child<ScalarNode>(1.0f, "dodging")
        .Composite(keep_facing ? std::unique_ptr<BehaviorNode>(std::make_unique<FaceNode>("aimshot"))
                               : std::unique_ptr<BehaviorNode>(std::make_unique<PassNode>()))
        .End();
  // clang-format on

  return builder.Build();
}

static std::unique_ptr<behavior::BehaviorNode> CreateEngage(const CombatSettings& settings);

// How far a bot bombing two other enemies' fight keeps from its target (tiles).
constexpr float kBattleDistance = 40.0f;

// Battle blasting (TargetLock): while bombing two other enemies' fight, keep well off it, facing it, unless the commit
// decision closes in to finish one.
static std::unique_ptr<behavior::BehaviorNode> CreateBattleMovement() {
  using namespace behavior;
  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Child<BlackboardSetQueryNode>("battle")
        .InvertChild<CombatModeIsNode>(CombatMode::Commit)
        .Child<FaceNode>("aimshot")
        .Child<SeekNode>("aimshot", kBattleDistance, SeekNode::DistanceResolveType::Dynamic)
        .End();
  // clang-format on

  return builder.Build();
}

// The long-range opening (OpeningCheckNode): straight at the target at full speed.
static std::unique_ptr<behavior::BehaviorNode> CreateOpening() {
  using namespace behavior;
  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Child<BlackboardSetQueryNode>("opening")
        .Child<FaceNode>("aimshot")
        .Child<SeekNode>("aimshot", 0.0f, SeekNode::DistanceResolveType::Static)
        .End();
  // clang-format on

  return builder.Build();
}

// Moving in a fight, by the commit decision's mode: commit closes in and finishes, disengage and recharge back off to
// the retreat distance, hold keeps the leash distance in range style, and pressure engages in the bot's own style.
static std::unique_ptr<behavior::BehaviorNode> CreateMovement(const CombatSettings& settings) {
  using namespace behavior;

  CombatSettings range = settings;
  range.engage_close = false;

  std::unique_ptr<BehaviorNode> back_off;
  if (settings.retreat_distance) {
    float distance = *settings.retreat_distance;
    back_off = std::make_unique<SeekNode>("aimshot", distance, SeekNode::DistanceResolveType::Dynamic);
  } else {
    back_off = std::make_unique<SeekNode>("aimshot", "leash_distance", SeekNode::DistanceResolveType::Dynamic);
  }

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Selector()
        .Sequence()
            .Child<CombatModeIsNode>(CombatMode::Commit)
            .Child<FaceNode>("aimshot")
            .Child<SeekNode>("aimshot", 0.0f, SeekNode::DistanceResolveType::Static)
            .End()
        .Sequence()
            .Child<CombatModeIsNode>(CombatMode::Disengage, CombatMode::Recharge)
            .Child<FaceNode>("aimshot")
            .Composite(std::move(back_off))
            .End()
        .Sequence()
            .Child<CombatModeIsNode>(CombatMode::Hold)
            .Composite(CreateEngage(range))
            .End()
        .Composite(CreateEngage(settings))
        .End();
  // clang-format on

  return builder.Build();
}

// Pressuring or holding: close in on the target, or hover around a territory position at the leash distance, facing
// the target only while not under threat.
static std::unique_ptr<behavior::BehaviorNode> CreateEngage(const CombatSettings& settings) {
  using namespace behavior;

  BehaviorBuilder builder;

  if (settings.engage_close) {
    // clang-format off
    builder
      .Sequence()
          .Child<FaceNode>("aimshot")
          .Child<SeekNode>("aimshot", 0.0f, SeekNode::DistanceResolveType::Zero)
          .End();
    // clang-format on

    return builder.Build();
  }

  // clang-format off
  builder
    .Selector()
        .Sequence()
            .Child<FindTerritoryPosition>("nearest_target", "leash_distance", "territory_position")
            .Sequence(CompositeDecorator::Success)
                .Child<PositionThreatQueryNode>("self_position", "self_threat", 8.0f, 3.0f)
                .Child<RenderTextNode>("ui_camera", Vector2f(512, 600), [](ExecuteContext& ctx) {
                  std::string str = std::string("Self threat: ") + std::to_string(ctx.blackboard.ValueOr<float>("self_threat", 0.0f));

                  return RenderTextNode::Request(str, TextColor::White, Layer::TopMost, TextAlignment::Center);
                })
                .Child<PositionThreatQueryNode>("territory_position", "territory_threat", 8.0f, 3.0f)
                .Child<RenderTextNode>("world_camera", "territory_position", [](ExecuteContext& ctx) {
                  std::string str = std::string("Threat: ") + std::to_string(ctx.blackboard.ValueOr<float>("territory_threat", 0.0f));

                  return RenderTextNode::Request(str, TextColor::White, Layer::TopMost, TextAlignment::Center);
                })
                .Child<ScalarThresholdNode<float>>("territory_threat", 0.2f)
                .Child<FindTerritoryPosition>("nearest_target", "leash_distance", "territory_position", true)
                .End()
            .Sequence(CompositeDecorator::Success)
                .InvertChild<ScalarThresholdNode<float>>("self_threat", 0.2f)
                .Child<FaceNode>("aimshot")
                .End()
            .Child<ArriveNode>("territory_position", 25.0f)
            .Child<RectangleNode>("territory_position", Vector2f(2.0f, 2.0f), "territory_rect")
            .Child<RenderRectNode>("world_camera", "territory_rect", Vector3f(0.0f, 1.0f, 0.0f))
            .End()
        .Sequence()
            .Child<FaceNode>("aimshot")
            .Child<SeekNode>("aimshot", "leash_distance")
            .End()
        .End();
  // clang-format on

  return builder.Build();
}

// Spread shots around targets beyond the range. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateShotSpread(const CombatSettings& settings) {
  using namespace behavior;
  if (!settings.shot_spread_range) return std::make_unique<PassNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Child<DistanceThresholdNode>("nearest_target_position", *settings.shot_spread_range)
        .Child<ShotSpreadNode>("aimshot", kShotSpread, kShotSpreadPeriod)
        .End();
  // clang-format on

  return std::make_unique<SuccessNode>(builder.Build());
}

// Aim at the target: as it is now, or with a reaction time as the bot saw it a moment ago; with full lead, past a
// wall it's about to bounce off.
static std::unique_ptr<behavior::BehaviorNode> CreateAim(const CombatSettings& settings, PlayerHistory& history) {
  using namespace behavior;
  std::unique_ptr<BehaviorNode> aim;
  if (settings.reaction_ticks <= 0 && settings.lead_skill == 1.0f) {
    aim = std::make_unique<AimNode>(WeaponType::Bullet, "nearest_target", "aimshot");
  } else {
    aim = std::make_unique<SkilledAimNode>(&history, "nearest_target", "aimshot", (u32)settings.reaction_ticks,
                                           settings.lead_skill);
  }
  if (settings.lead_skill < 1.0f) return aim;

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Composite(std::move(aim))
        .Child<ReboundAimNode>(&history, (u32)settings.reaction_ticks, "nearest_target", "aimshot")
        .End();
  // clang-format on

  return builder.Build();
}

// Aim drift, when the skill has any. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateAimError(const CombatSettings& settings) {
  if (settings.aim_error <= 0.0f) return std::make_unique<PassNode>();

  return std::make_unique<AimErrorNode>("aimshot", settings.aim_error);
}

// Whether a shot is worth taking, by expected value (which also keeps the energy a backing off or recharging bot
// needs); while pursued, a cover shot is always worth it when the bot uses cover fire.
static std::unique_ptr<behavior::BehaviorNode> CreateShotChoice(const CombatSettings& settings, WeaponType weapon,
                                                                const char* velocity_key, Opportunities& opportunities) {
  using namespace behavior;
  auto value = [&]() {
    return std::make_unique<ShotValueNode>(weapon, velocity_key, settings.shot_judgement, settings.bomb_appetite,
                                           settings.lead_skill, &opportunities);
  };
  if (!settings.cover_fire) return value();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Selector()
        .Child<CoverShotNode>()
        .Composite(value())
        .End();
  // clang-format on

  return builder.Build();
}

// Accepting hits, when the bot's risk appetite allows any. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateAccept(const CombatSettings& settings, CommitDecision& decision) {
  if (settings.accept_confidence > 1.0f - 1e-3f) return std::make_unique<PassNode>();

  return std::make_unique<AcceptNode>(&decision, settings.accept_confidence, settings.dodge_threshold.value_or(1.0f),
                                      settings.damage_judgement, settings.judgement_noise);
}

// Following the patrol route: today's way (face each waypoint, arrive, switch within 15 tiles), or a RouteNode style.
static std::unique_ptr<behavior::BehaviorNode> CreatePatrol(const CombatSettings& settings) {
  using namespace behavior;
  BehaviorBuilder builder;

  if (settings.navigation == (int)Navigation::Classic) {
    // clang-format off
    builder
      .Sequence() // Follow set waypoints.
          .Child<WaypointNode>("waypoints", "waypoint_index", "waypoint_position", 15.0f)
          .Selector()
              .Sequence()
                  .InvertChild<VisibilityQueryNode>("waypoint_position")
                  .Child<GoToNode>("waypoint_position")
                  .Child<RenderPathNode>(Vector3f(0.0f, 0.5f, 1.0f))
                  .End()
              .Parallel()
                  .Child<FaceNode>("waypoint_position")
                  .Child<ArriveNode>("waypoint_position", 1.25f)
                  .End()
              .End()
          .End();
    // clang-format on
    return builder.Build();
  }

  // clang-format off
  builder
    .Selector() // Follow set waypoints, pathing round walls when the way ahead isn't clear.
        .Child<RouteNode>((Navigation)settings.navigation, settings.look_ahead, settings.corner_speed)
        .Sequence()
            .Child<GoToNode>("waypoint_position")
            .Child<RenderPathNode>(Vector3f(0.0f, 0.5f, 1.0f))
            .End()
        .End();
  // clang-format on
  return builder.Build();
}

// Whether the bot is pursued while backing off, for cover shots and mines.
static std::unique_ptr<behavior::BehaviorNode> CreatePursuitCheck(const CombatSettings& settings) {
  if (!settings.cover_fire && !settings.mine_cover) return std::make_unique<PassNode>();

  return std::make_unique<PursuitNode>();
}

// Mines in a pursuer's path, when the bot uses them. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateMineCover(const CombatSettings& settings) {
  if (!settings.mine_cover) return std::make_unique<PassNode>();

  // A bot with good judgement waits until the pursuer is on its trail.
  return std::make_unique<behavior::SuccessNode>(
      std::make_unique<MineCoverNode>(settings.mine_cover_range, settings.mine_on_trail));
}

// Bomb only into enough of our own bullet damage already heading at the target. Succeeds when that's the case, or
// when the settings don't ask for it.
static std::unique_ptr<behavior::BehaviorNode> CreateBombOverlapCheck(const CombatSettings& settings) {
  using namespace behavior;
  if (!settings.bomb_overlap_damage) return std::make_unique<PassNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Child<IncomingDamageQueryNode>("nearest_target", kOutgoingDamageRadius, 2.75f, "outgoing_damage")
        .Child<ScalarThresholdNode<float>>("outgoing_damage", *settings.bomb_overlap_damage)
        .End();
  // clang-format on

  return builder.Build();
}

std::unique_ptr<behavior::BehaviorNode> CreateCombatTree(const CombatSettings& settings, CombatSenses senses) {
  using namespace behavior;

  BehaviorBuilder builder;

  // The built-in numbers are the SVS Warbird tree's.
  const float multifire_range = settings.multifire_range.value_or(15.0f);
  const float antiwarp_energy = settings.antiwarp_energy.value_or(0.75f);
  const float repel_trigger = settings.repel_trigger.value_or(1.0f);
  const float bomb_min_range = settings.bomb_min_range.value_or(kDefaultBombMinRange);

  // clang-format off
  builder
    .Selector()
        .Sequence() // Enter the specified ship if not already in it.
            .InvertChild<ShipQueryNode>("request_ship")
            .Child<ShipRequestNode>("request_ship")
            .End()
        .Selector() // Choose to fight the player or follow waypoints.
            .Sequence() // Find nearest target and either path to them or seek them directly.
                .Sequence()
                    .Child<PlayerPositionQueryNode>("self_position")
                    .Child<LockedTargetNode>(&senses.target_lock, "nearest_target")
                    .Child<PlayerPositionQueryNode>("nearest_target", "nearest_target_position")
                    .Child<EnergyEstimateQueryNode>(&senses.estimate, "nearest_target", "nearest_target_energy", "nearest_target_energy_sd")
                    .Child<CommitDecisionNode>(&senses.decision, "nearest_target")
                    .End()
                .Sequence(CompositeDecorator::Success) // If we have a portal but no location, lay one down.
                    .Child<ShipItemCountThresholdNode>(ShipItemType::Portal, 1)
                    .InvertChild<ShipPortalPositionQueryNode>()
                    .Child<InputActionNode>(InputAction::Portal)
                    .End()
                .Selector(CompositeDecorator::Success) // Multifire on when far from the enemy, off when close.
                    .Sequence()
                        .Child<ShipCapabilityQueryNode>(ShipCapability_Multifire)
                        .Child<DistanceThresholdNode>("nearest_target_position", multifire_range)
                        .InvertChild<ShipMultifireQueryNode>()
                        .Child<InputActionNode>(InputAction::Multifire)
                        .End()
                    .Sequence()
                        .Child<ShipCapabilityQueryNode>(ShipCapability_Multifire)
                        .InvertChild<DistanceThresholdNode>("nearest_target_position", multifire_range)
                        .Child<ShipMultifireQueryNode>()
                        .Child<InputActionNode>(InputAction::Multifire)
                        .End()
                    .End()
                .Selector(CompositeDecorator::Success) // Toggle antiwarp based on energy
                    .Sequence() // Enable antiwarp if we are healthy
                        .Child<ShipCapabilityQueryNode>(ShipCapability_Antiwarp)
                        .Child<PlayerEnergyPercentThresholdNode>(antiwarp_energy)
                        .InvertChild<PlayerStatusQueryNode>(Status_Antiwarp)
                        .Child<InputActionNode>(InputAction::Antiwarp)
                        .End()
                    .Sequence() // Disable antiwarp if we aren't healthy
                        .Child<ShipCapabilityQueryNode>(ShipCapability_Antiwarp)
                        .InvertChild<PlayerEnergyPercentThresholdNode>(antiwarp_energy)
                        .Child<PlayerStatusQueryNode>(Status_Antiwarp)
                        .Child<InputActionNode>(InputAction::Antiwarp)
                        .End()
                    .End()
                .Sequence(CompositeDecorator::Success) // Use defensive items.
                    .Sequence(CompositeDecorator::Success) // Always check incoming damage so we can use it in repel and portal sequences.
                        .Child<RepelDistanceQueryNode>("repel_distance")
                        .Child<IncomingDamageQueryNode>("repel_distance", "incoming_damage")
                        .Child<PlayerCurrentEnergyQueryNode>("self_energy")
                        .Child<ScaleNode>("self_energy", repel_trigger, "repel_energy")
                        .End()
                    .Composite(CreateAccept(settings, senses.decision))
                    .Sequence(CompositeDecorator::Success) // If we are in danger but can't repel, use our portal.
                        .InvertChild<BlackboardSetQueryNode>("accepting")
                        .InvertChild<ShipItemCountThresholdNode>(ShipItemType::Repel)
                        .Child<ShipPortalPositionQueryNode>() // Check if we have a portal down.
                        .Child<ScalarThresholdNode<float>>("incoming_damage", "repel_energy")
                        .Child<TimerExpiredNode>("defense_timer")
                        .Child<InputActionNode>(InputAction::Warp)
                        .Child<TimerSetNode>("defense_timer", 100)
                        .End()
                    .Sequence(CompositeDecorator::Success) // Use repel when in danger.
                        .InvertChild<BlackboardSetQueryNode>("accepting")
                        .Child<ShipWeaponCapabilityQueryNode>(WeaponType::Repel)
                        .Child<TimerExpiredNode>("defense_timer")
                        .Child<ScalarThresholdNode<float>>("incoming_damage", "repel_energy")
                        .Child<InputActionNode>(InputAction::Repel)
                        .Child<TimerSetNode>("defense_timer", 100)
                        .End()
                    .End()
                .Selector()
                    .Sequence() // Path to target if they aren't immediately visible, dodging on the way.
                        .InvertChild<VisibilityQueryNode>("nearest_target_position")
                        .Selector()
                            .Composite(CreateDodge(settings, false, senses.weapon_notice))
                            .Sequence()
                                .Child<GoToNode>("nearest_target_position")
                                .Child<RenderPathNode>(Vector3f(0.0f, 1.0f, 0.5f))
                                .End()
                            .End()
                        .End()
                    .Sequence() // Aim at target and shoot while moving.
                        .Composite(CreateAim(settings, senses.history))
                        .Composite(CreateShotSpread(settings))
                        .Composite(CreateAimError(settings))
                        .Composite(CreatePursuitCheck(settings))
                        .Child<OpeningCheckNode>(&senses.opportunities, settings.dodge_threshold)
                        .Child<ThrustTimingNode>(settings.piloting)
                        .Parallel()
                            .Child<BlackboardEraseNode>("dodging")
                            .Selector() // Moving: dodge, battle blasting, an opening, or move by the mode.
                                .Composite(CreateDodge(settings, true, senses.weapon_notice))
                                .Composite(CreateBattleMovement())
                                .Composite(CreateOpening())
                                .Composite(CreateMovement(settings))
                                .End()
                            .Composite(CreateFlightPhases(settings, senses.weapon_notice))
                            .Sequence(CompositeDecorator::Success) // Use burst when near a wall.
                                .Child<ShipWeaponCapabilityQueryNode>(WeaponType::Burst)
                                .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bomb)
                                .InvertChild<DistanceThresholdNode>("nearest_target_position", 15.0f)
                                .Child<BurstAreaQueryNode>()
                                .Child<InputActionNode>(InputAction::Burst)
                                .End()
                            .Sequence(CompositeDecorator::Success) // Bomb fire check.
                                .Child<ShipWeaponCapabilityQueryNode>(WeaponType::Bomb)
                                .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bomb)
                                .Composite(CreateBombOverlapCheck(settings))
                                .Child<DistanceThresholdNode>("nearest_target_position", bomb_min_range)
                                .Child<ShotVelocityQueryNode>(WeaponType::Bomb, "bomb_fire_velocity")
                                .Composite(CreateShotChoice(settings, WeaponType::Bomb, "bomb_fire_velocity", senses.opportunities))
                                .Child<RayNode>("self_position", "bomb_fire_velocity", "bomb_fire_ray")
                                .Child<DynamicPlayerBoundingBoxQueryNode>("nearest_target", "target_bounds", kBombBoxScale)
                                .Child<MoveRectangleNode>("target_bounds", "aimshot", "target_bounds")
                                .Child<RenderRectNode>("world_camera", "target_bounds", Vector3f(1.0f, 0.0f, 0.0f))
                                .Child<RenderRayNode>("world_camera", "bomb_fire_ray", 50.0f, Vector3f(1.0f, 1.0f, 0.0f))
                                .Child<RayRectangleInterceptNode>("bomb_fire_ray", "target_bounds")
                                .Child<InputActionNode>(InputAction::Bomb)
                                .End()
                            .Sequence(CompositeDecorator::Success) // Determine if a shot should be fired by using weapon trajectory and bounding boxes.
                                .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bullet)
                                .InvertChild<InputQueryNode>(InputAction::Bomb) // Don't try to shoot a bullet when shooting a bomb.
                                .InvertChild<TileQueryNode>(kTileIdSafe)
                                .Child<ShotVelocityQueryNode>(WeaponType::Bullet, "bullet_fire_velocity")
                                .Composite(CreateShotChoice(settings, WeaponType::Bullet, "bullet_fire_velocity", senses.opportunities))
                                .Child<RayNode>("self_position", "bullet_fire_velocity", "bullet_fire_ray")
                                .Child<DynamicPlayerBoundingBoxQueryNode>("nearest_target", "target_bounds", kBulletBoxScale)
                                .Child<MoveRectangleNode>("target_bounds", "aimshot", "target_bounds")
                                .Child<RayRectangleInterceptNode>("bullet_fire_ray", "target_bounds")
                                .Child<InputActionNode>(InputAction::Bullet)
                                .End()
                            .Composite(CreateMineCover(settings))
                            .End()
                        .End()
                    .End()
                .End()
            .Sequence()
                .Child<FindNearestGreenNode>("nearest_green_position")
                .Child<GoToNode>("nearest_green_position")
                .Child<RenderPathNode>(Vector3f(0.0f, 1.0f, 0.0f))
                .End()
            .Composite(CreatePatrol(settings))
            .End()
        .End();
  // clang-format on

  return builder.Build();
}

}  // namespace chaos
}  // namespace zero
