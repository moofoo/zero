#include "ChaosCombat.h"

#include "CommitDecision.h"
#include "EnergyEstimate.h"
#include "PlayerHistory.h"
#include "TargetLock.h"

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

// How close a low-energy enemy has to be for a bot that rushes to stop dodging and finish them off (tiles).
constexpr float kNearbyEnemyThreshold = 20.0f;
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
  CommitDecisionNode(CombatDecision* decision, const char* target_key) : decision(decision), target_key(target_key) {}

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

  CombatDecision* decision;
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
    if (!target) return behavior::ExecuteResult::Failure;

    ctx.blackboard.Set(player_key, target);
    return behavior::ExecuteResult::Success;
  }

  TargetLock* lock;
  const char* player_key;
  std::vector<Player*> candidates;
};

// Aims at the target as the bot saw it `delay` ticks ago: a player's reaction time.
struct PerceivedAimNode : public behavior::BehaviorNode {
  PerceivedAimNode(PlayerHistory* history, const char* target_key, const char* position_key, u32 delay)
      : history(history), target_key(target_key), position_key(position_key), delay(delay) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_target = ctx.blackboard.Value<Player*>(target_key);
    if (!self || !opt_target || !*opt_target) return behavior::ExecuteResult::Failure;

    PlayerHistory::Sample seen = history->Get(**opt_target, delay);
    Vector2f aimshot = behavior::CalculateAimshot(*ctx.bot->game, *self, WeaponType::Bullet, seen.position,
                                                  seen.velocity);
    ctx.blackboard.Set(position_key, aimshot);

    return behavior::ExecuteResult::Success;
  }

  PlayerHistory* history;
  const char* target_key;
  const char* position_key;
  u32 delay;
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

// Grows or shrinks a box about its centre.
struct ScaleBoundsNode : public behavior::BehaviorNode {
  ScaleBoundsNode(const char* bounds_key, float scale) : bounds_key(bounds_key), scale(scale) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    auto opt_bounds = ctx.blackboard.Value<Rectangle>(bounds_key);
    if (!opt_bounds) return behavior::ExecuteResult::Failure;

    Vector2f center = (opt_bounds->min + opt_bounds->max) * 0.5f;
    Vector2f half = (opt_bounds->max - opt_bounds->min) * (0.5f * scale);
    ctx.blackboard.Set(bounds_key, Rectangle(center - half, center + half));

    return behavior::ExecuteResult::Success;
  }

  const char* bounds_key;
  float scale;
};

// Succeeds when a shot's velocity carries it towards the aim point at least this fast (tiles/s): slower shots are
// easy to dodge.
struct MinShotSpeedNode : public behavior::BehaviorNode {
  MinShotSpeedNode(const char* velocity_key, const char* position_key, float min_speed)
      : velocity_key(velocity_key), position_key(position_key), min_speed(min_speed) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    auto opt_velocity = ctx.blackboard.Value<Vector2f>(velocity_key);
    auto opt_position = ctx.blackboard.Value<Vector2f>(position_key);
    if (!self || !opt_velocity || !opt_position) return behavior::ExecuteResult::Failure;

    float speed = opt_velocity->Dot(Normalize(*opt_position - self->position));
    return speed >= min_speed ? behavior::ExecuteResult::Success : behavior::ExecuteResult::Failure;
  }

  const char* velocity_key;
  const char* position_key;
  float min_speed;
};

// Dodge when incoming damage is at least the threshold fraction of max energy, unless a low-energy enemy we'd rush
// is close by (then stay on them). Dodging only steers: while fighting, the bot keeps facing its aim point, and aiming
// and firing carry on, as a player dodges and shoots at once.
static std::unique_ptr<behavior::BehaviorNode> CreateDodge(const CombatSettings& settings, bool keep_facing) {
  using namespace behavior;
  if (!settings.dodge_threshold) return std::make_unique<FailureNode>();

  // Without a rush threshold, no enemy is low enough to stay on.
  float rush_enemy_energy = settings.rush_enemy_energy.value_or(0.0f);
  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Sequence(CompositeDecorator::Invert)
            .InvertChild<ScalarThresholdNode<float>>("nearest_target_energy", rush_enemy_energy)
            .InvertChild<DistanceThresholdNode>("nearest_target_position", "self_position", kNearbyEnemyThreshold)
            .End()
        .Child<DodgeIncomingDamage>(*settings.dodge_threshold, 35.0f)
        .Composite(keep_facing ? std::unique_ptr<BehaviorNode>(std::make_unique<FaceNode>("aimshot"))
                               : std::unique_ptr<BehaviorNode>(std::make_unique<PassNode>()))
        .End();
  // clang-format on

  return builder.Build();
}

// Rush an enemy whose energy is below the threshold. While rushing, bullets ignore the energy floor.
static std::unique_ptr<behavior::BehaviorNode> CreateRush(const CombatSettings& settings) {
  using namespace behavior;
  if (!settings.rush_enemy_energy) return std::make_unique<FailureNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .InvertChild<ScalarThresholdNode<float>>("nearest_target_energy", *settings.rush_enemy_energy)
        .Child<FaceNode>("aimshot")
        .Child<SeekNode>("aimshot", 0.0f, SeekNode::DistanceResolveType::Static)
        .Child<ScalarNode>(1.0f, "rushing")
        .End();
  // clang-format on

  return builder.Build();
}

// Back off below the energy fraction: to the retreat distance, further than the bot hovers, or without one to the
// leash distance.
static std::unique_ptr<behavior::BehaviorNode> CreateRetreat(const CombatSettings& settings) {
  using namespace behavior;
  if (!settings.retreat_energy) return std::make_unique<FailureNode>();

  std::unique_ptr<BehaviorNode> seek;
  if (settings.retreat_distance) {
    seek = std::make_unique<SeekNode>("aimshot", *settings.retreat_distance, SeekNode::DistanceResolveType::Dynamic);
  } else {
    seek = std::make_unique<SeekNode>("aimshot", "leash_distance", SeekNode::DistanceResolveType::Dynamic);
  }

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .InvertChild<PlayerEnergyPercentThresholdNode>(*settings.retreat_energy)
        .Child<FaceNode>("aimshot")
        .Composite(std::move(seek))
        .Child<ScalarNode>(1.0f, "backing_off")
        .End();
  // clang-format on

  return builder.Build();
}

static std::unique_ptr<behavior::BehaviorNode> CreateEngage(const CombatSettings& settings);

// Moving by the commit decision's mode: commit closes in and finishes (bullets ignore the energy floor, as in a rush),
// disengage and recharge back off to the retreat distance, hold keeps the leash distance in range style, and pressure
// engages in the bot's own style.
static std::unique_ptr<behavior::BehaviorNode> CreateModeMovement(const CombatSettings& settings) {
  using namespace behavior;

  CombatSettings range = settings;
  range.engage_close = false;

  std::unique_ptr<BehaviorNode> back_off;
  if (settings.retreat_distance) {
    back_off = std::make_unique<SeekNode>("aimshot", *settings.retreat_distance, SeekNode::DistanceResolveType::Dynamic);
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
            .Child<ScalarNode>(1.0f, "rushing")
            .End()
        .Sequence()
            .Child<CombatModeIsNode>(CombatMode::Disengage, CombatMode::Recharge)
            .Child<FaceNode>("aimshot")
            .Composite(std::move(back_off))
            .Child<ScalarNode>(1.0f, "backing_off")
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

// Moving when not rushing or backing off: close in on the target, or hover around a territory position at the
// leash distance, facing the target only while not under threat.
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

// Moving in a fight: by the commit decision when it's on, else rush, back off or engage by their triggers.
static std::unique_ptr<behavior::BehaviorNode> CreateMovement(const CombatSettings& settings) {
  using namespace behavior;
  if (settings.commit_decision) return CreateModeMovement(settings);

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Selector()
        .Composite(CreateRush(settings))
        .Composite(CreateRetreat(settings))
        .Composite(CreateEngage(settings))
        .End();
  // clang-format on

  return builder.Build();
}

// The commit decision, when it's on. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateDecision(const CombatSettings& settings, CombatDecision& decision) {
  if (!settings.commit_decision) return std::make_unique<PassNode>();

  return std::make_unique<CommitDecisionNode>(&decision, "nearest_target");
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

// Aim at the target: as it is now, or with a reaction time as the bot saw it a moment ago.
static std::unique_ptr<behavior::BehaviorNode> CreateAim(const CombatSettings& settings, PlayerHistory& history) {
  using namespace behavior;
  if (settings.reaction_ticks <= 0) return std::make_unique<AimNode>(WeaponType::Bullet, "nearest_target", "aimshot");

  return std::make_unique<PerceivedAimNode>(&history, "nearest_target", "aimshot", (u32)settings.reaction_ticks);
}

// Aim drift, when the skill has any. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateAimError(const CombatSettings& settings) {
  if (settings.aim_error <= 0.0f) return std::make_unique<PassNode>();

  return std::make_unique<AimErrorNode>("aimshot", settings.aim_error);
}

// Loosens or tightens the box a shot must cross, when the skill scales it. Always succeeds.
static std::unique_ptr<behavior::BehaviorNode> CreateHitBoxScale(const CombatSettings& settings) {
  if (settings.hit_box_scale == 1.0f) return std::make_unique<PassNode>();

  return std::make_unique<ScaleBoundsNode>("target_bounds", settings.hit_box_scale);
}

// Holds shots too slow to land, when the skill asks, except while backing off: slow shots dropped in a pursuer's
// path are cover, not a wasted attack.
static std::unique_ptr<behavior::BehaviorNode> CreateShotSpeedCheck(const CombatSettings& settings,
                                                                    const char* velocity_key) {
  using namespace behavior;
  if (settings.min_shot_speed <= 0.0f) return std::make_unique<PassNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Selector()
        .Child<BlackboardSetQueryNode>("backing_off")
        .Child<MinShotSpeedNode>(velocity_key, "aimshot", settings.min_shot_speed)
        .End();
  // clang-format on

  return builder.Build();
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
  const float bomb_energy = settings.bomb_energy.value_or(0.65f);
  const float bomb_min_range = settings.bomb_min_range.value_or(10.0f);
  const float bullet_energy = settings.bullet_energy.value_or(0.3f);

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
                    .Composite(CreateDecision(settings, senses.decision))
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
                    .Sequence(CompositeDecorator::Success) // If we are in danger but can't repel, use our portal.
                        .InvertChild<ShipItemCountThresholdNode>(ShipItemType::Repel)
                        .Child<ShipPortalPositionQueryNode>() // Check if we have a portal down.
                        .Child<ScalarThresholdNode<float>>("incoming_damage", "repel_energy")
                        .Child<TimerExpiredNode>("defense_timer")
                        .Child<InputActionNode>(InputAction::Warp)
                        .Child<TimerSetNode>("defense_timer", 100)
                        .End()
                    .Sequence(CompositeDecorator::Success) // Use repel when in danger.
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
                            .Composite(CreateDodge(settings, false))
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
                        .Parallel()
                            .Child<BlackboardEraseNode>("rushing")
                            .Child<BlackboardEraseNode>("backing_off")
                            .Selector() // Moving: dodge, rush, back off, or engage.
                                .Composite(CreateDodge(settings, true))
                                .Composite(CreateMovement(settings))
                                .End()
                            .Sequence(CompositeDecorator::Success) // Use burst when near a wall.
                                .Child<ShipWeaponCapabilityQueryNode>(WeaponType::Burst)
                                .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bomb)
                                .InvertChild<DistanceThresholdNode>("nearest_target_position", 15.0f)
                                .Child<BurstAreaQueryNode>()
                                .Child<InputActionNode>(InputAction::Burst)
                                .End()
                            .Sequence(CompositeDecorator::Success) // Bomb fire check.
                                .Child<PlayerEnergyPercentThresholdNode>(bomb_energy)
                                .Child<ShipWeaponCapabilityQueryNode>(WeaponType::Bomb)
                                .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bomb)
                                .Composite(CreateBombOverlapCheck(settings))
                                .Child<DistanceThresholdNode>("nearest_target_position", bomb_min_range)
                                .Child<ShotVelocityQueryNode>(WeaponType::Bomb, "bomb_fire_velocity")
                                .Composite(CreateShotSpeedCheck(settings, "bomb_fire_velocity"))
                                .Child<RayNode>("self_position", "bomb_fire_velocity", "bomb_fire_ray")
                                .Child<DynamicPlayerBoundingBoxQueryNode>("nearest_target", "target_bounds", 3.0f)
                                .Composite(CreateHitBoxScale(settings))
                                .Child<MoveRectangleNode>("target_bounds", "aimshot", "target_bounds")
                                .Child<RenderRectNode>("world_camera", "target_bounds", Vector3f(1.0f, 0.0f, 0.0f))
                                .Child<RenderRayNode>("world_camera", "bomb_fire_ray", 50.0f, Vector3f(1.0f, 1.0f, 0.0f))
                                .Child<RayRectangleInterceptNode>("bomb_fire_ray", "target_bounds")
                                .Child<InputActionNode>(InputAction::Bomb)
                                .End()
                            .Sequence(CompositeDecorator::Success) // Determine if a shot should be fired by using weapon trajectory and bounding boxes.
                                .Selector() // Above the energy floor, or rushing.
                                    .Child<BlackboardSetQueryNode>("rushing")
                                    .Child<PlayerEnergyPercentThresholdNode>(bullet_energy)
                                    .End()
                                .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bullet)
                                .InvertChild<InputQueryNode>(InputAction::Bomb) // Don't try to shoot a bullet when shooting a bomb.
                                .InvertChild<TileQueryNode>(kTileIdSafe)
                                .Child<ShotVelocityQueryNode>(WeaponType::Bullet, "bullet_fire_velocity")
                                .Composite(CreateShotSpeedCheck(settings, "bullet_fire_velocity"))
                                .Child<RayNode>("self_position", "bullet_fire_velocity", "bullet_fire_ray")
                                .Child<DynamicPlayerBoundingBoxQueryNode>("nearest_target", "target_bounds", 4.0f)
                                .Composite(CreateHitBoxScale(settings))
                                .Child<MoveRectangleNode>("target_bounds", "aimshot", "target_bounds")
                                .Child<RayRectangleInterceptNode>("bullet_fire_ray", "target_bounds")
                                .Child<InputActionNode>(InputAction::Bullet)
                                .End()
                            .End()
                        .End()
                    .End()
                .End()
            .Sequence()
                .Child<FindNearestGreenNode>("nearest_green_position")
                .Child<GoToNode>("nearest_green_position")
                .Child<RenderPathNode>(Vector3f(0.0f, 1.0f, 0.0f))
                .End()
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
                .End()
            .End()
        .End();
  // clang-format on

  return builder.Build();
}

}  // namespace chaos
}  // namespace zero
