#include "WarbirdBehavior.h"

#include <zero/behavior/BehaviorBuilder.h>
#include <zero/behavior/BehaviorTree.h>
#include <zero/behavior/nodes/AimNode.h>
#include <zero/behavior/nodes/BlackboardNode.h>
#include <zero/behavior/nodes/InputActionNode.h>
#include <zero/behavior/nodes/MapNode.h>
#include <zero/behavior/nodes/MathNode.h>
#include <zero/behavior/nodes/MoveNode.h>
#include <zero/behavior/nodes/PlayerNode.h>
#include <zero/behavior/nodes/PowerballNode.h>
#include <zero/behavior/nodes/RegionNode.h>
#include <zero/behavior/nodes/RenderNode.h>
#include <zero/behavior/nodes/ShipNode.h>
#include <zero/behavior/nodes/ThreatNode.h>
#include <zero/behavior/nodes/TimerNode.h>
#include <zero/behavior/nodes/WaypointNode.h>
#include <zero/zones/svs/nodes/BurstAreaQueryNode.h>
#include <zero/zones/svs/nodes/DynamicPlayerBoundingBoxQueryNode.h>
#include <zero/zones/svs/nodes/FindNearestGreenNode.h>
#include <zero/zones/svs/nodes/IncomingDamageQueryNode.h>
#include <zero/zones/svs/nodes/MemoryTargetNode.h>
#include <zero/zones/svs/nodes/NearbyEnemyWeaponQueryNode.h>

namespace zero {
namespace svs {

// The Terrier's dodge, which an aggression preset can give the Warbird: dodge when incoming damage is at least the
// threshold fraction of max energy, unless a low-energy enemy is close by (then stay on them).
static std::unique_ptr<behavior::BehaviorNode> CreateDodge(const AggressionSettings& aggression) {
  using namespace behavior;
  if (!aggression.dodge_threshold) return std::make_unique<FailureNode>();

  // Without a rush threshold, no enemy is low enough to stay on.
  float rush_enemy_energy = aggression.rush_enemy_energy.value_or(0.0f);
  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Sequence(CompositeDecorator::Invert)
            .Child<PlayerEnergyQueryNode>("nearest_target", "nearest_target_energy")
            .InvertChild<ScalarThresholdNode<float>>("nearest_target_energy", rush_enemy_energy)
            .InvertChild<DistanceThresholdNode>("nearest_target_position", "self_position", 20.0f)
            .End()
        .Child<DodgeIncomingDamage>(*aggression.dodge_threshold, 35.0f)
        .End();
  // clang-format on

  return builder.Build();
}

// Rushing an enemy whose energy is below the preset's threshold, as the Terrier does.
static std::unique_ptr<behavior::BehaviorNode> CreateRush(const AggressionSettings& aggression) {
  using namespace behavior;
  if (!aggression.rush_enemy_energy) return std::make_unique<FailureNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .Child<PlayerEnergyQueryNode>("nearest_target", "nearest_target_energy")
        .InvertChild<ScalarThresholdNode<float>>("nearest_target_energy", *aggression.rush_enemy_energy)
        .Child<FaceNode>("aimshot")
        .Child<SeekNode>("aimshot", 0.0f, SeekNode::DistanceResolveType::Static)
        .End();
  // clang-format on

  return builder.Build();
}

// Backing off to the leash distance below the preset's energy fraction, as the Terrier does.
static std::unique_ptr<behavior::BehaviorNode> CreateRetreat(const AggressionSettings& aggression) {
  using namespace behavior;
  if (!aggression.retreat_energy) return std::make_unique<FailureNode>();

  BehaviorBuilder builder;

  // clang-format off
  builder
    .Sequence()
        .InvertChild<PlayerEnergyPercentThresholdNode>(*aggression.retreat_energy)
        .Child<FaceNode>("aimshot")
        .Child<SeekNode>("aimshot", "leash_distance", SeekNode::DistanceResolveType::Dynamic)
        .End();
  // clang-format on

  return builder.Build();
}

std::unique_ptr<behavior::BehaviorNode> WarbirdBehavior::CreateTree(behavior::ExecuteContext& ctx) {
  using namespace behavior;

  BehaviorBuilder builder;

  const Vector2f center(512, 512);
  // Thresholds an aggression preset can change; the defaults are this tree's own numbers.
  const float multifire_range = aggression.multifire_range.value_or(15.0f);
  const float antiwarp_energy = aggression.antiwarp_energy.value_or(0.75f);
  const float repel_trigger = aggression.repel_trigger.value_or(1.0f);
  const float bomb_energy = aggression.bomb_energy.value_or(0.65f);
  const float bomb_min_range = aggression.bomb_min_range.value_or(10.0f);
  const float bullet_energy = aggression.bullet_energy.value_or(0.3f);

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
                    .Child<NearestMemoryTargetNode>("nearest_target")
                    .Child<PlayerPositionQueryNode>("nearest_target", "nearest_target_position")
                    .End()
                .Sequence(CompositeDecorator::Success) // If we have a portal but no location, lay one down.
                    .Child<ShipItemCountThresholdNode>(ShipItemType::Portal, 1)
                    .InvertChild<ShipPortalPositionQueryNode>()
                    .Child<InputActionNode>(InputAction::Portal)
                    .End()
                .Selector(CompositeDecorator::Success) // Enable multifire if ship supports it and it's disabled.
                    .Sequence()
                        .Child<ShipCapabilityQueryNode>(ShipCapability_Multifire)
                        .Child<DistanceThresholdNode>("nearest_target_position", multifire_range) // If we are far from enemy, use multifire
                        .InvertChild<ShipMultifireQueryNode>()
                        .Child<InputActionNode>(InputAction::Multifire)
                        .End()
                    .Sequence()
                        .Child<ShipCapabilityQueryNode>(ShipCapability_Multifire)
                        .InvertChild<DistanceThresholdNode>("nearest_target_position", multifire_range) // If we are far from enemy, use multifire
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
                .Selector()
                    .Composite(CreateDodge(aggression)) // Only with an aggression preset that sets a dodge threshold.
                    .Sequence() // Path to target if they aren't immediately visible.
                        .InvertChild<VisibilityQueryNode>("nearest_target_position")
                        .Child<GoToNode>("nearest_target_position")
                        .Child<RenderPathNode>(Vector3f(0.0f, 1.0f, 0.5f))
                        .End()
                    .Sequence() // Aim at target and shoot while seeking them.
                        .Child<AimNode>(WeaponType::Bullet, "nearest_target", "aimshot")
                        .Parallel()
                            .Selector() // Select between hovering around a territory position and seeking to enemy.
                                .Composite(CreateRush(aggression)) // Only with an aggression preset that sets these.
                                .Composite(CreateRetreat(aggression))
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
                                .End()
                            .Parallel()
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
                                    .Child<DistanceThresholdNode>("nearest_target_position", bomb_min_range)
                                    .Child<ShotVelocityQueryNode>(WeaponType::Bomb, "bomb_fire_velocity")
                                    .Child<RayNode>("self_position", "bomb_fire_velocity", "bomb_fire_ray")
                                    .Child<DynamicPlayerBoundingBoxQueryNode>("nearest_target", "target_bounds", 3.0f)
                                    .Child<MoveRectangleNode>("target_bounds", "aimshot", "target_bounds")
                                    .Child<RenderRectNode>("world_camera", "target_bounds", Vector3f(1.0f, 0.0f, 0.0f))
                                    .Child<RenderRayNode>("world_camera", "bomb_fire_ray", 50.0f, Vector3f(1.0f, 1.0f, 0.0f))
                                    .Child<RayRectangleInterceptNode>("bomb_fire_ray", "target_bounds")
                                    .Child<InputActionNode>(InputAction::Bomb)
                                    .End()
                                .Sequence(CompositeDecorator::Success) // Determine if a shot should be fired by using weapon trajectory and bounding boxes.
                                    .Child<PlayerEnergyPercentThresholdNode>(bullet_energy)
                                    .InvertChild<ShipWeaponCooldownQueryNode>(WeaponType::Bullet)
                                    .InvertChild<InputQueryNode>(InputAction::Bomb) // Don't try to shoot a bullet when shooting a bomb.
                                    .InvertChild<TileQueryNode>(kTileIdSafe)
                                    .Child<ShotVelocityQueryNode>(WeaponType::Bullet, "bullet_fire_velocity")
                                    .Child<RayNode>("self_position", "bullet_fire_velocity", "bullet_fire_ray")
                                    .Child<DynamicPlayerBoundingBoxQueryNode>("nearest_target", "target_bounds", 4.0f)
                                    .Child<MoveRectangleNode>("target_bounds", "aimshot", "target_bounds")
                                    .Child<RayRectangleInterceptNode>("bullet_fire_ray", "target_bounds")
                                    .Child<InputActionNode>(InputAction::Bullet)
                                    .End()
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

}  // namespace svs
}  // namespace zero
