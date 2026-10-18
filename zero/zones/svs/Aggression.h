#pragma once

#include <zero/behavior/BehaviorTree.h>

#include <optional>

namespace zero {
namespace svs {

// Tactical thresholds an aggression preset can change. Chaos reads them from an [Aggression.<name>] config section
// chosen with --aggression. A value left unset keeps the behavior's built-in number, and a branch a behavior doesn't
// have (the Warbird's retreat, rush and dodge) is only added when its value is set, so without a preset, and in the
// SVS zone, the trees are unchanged.
struct AggressionSettings {
  // Fire bombs / bullets only above this fraction of max energy.
  std::optional<float> bomb_energy;
  std::optional<float> bullet_energy;
  // Back off to the leash distance below this fraction of max energy.
  std::optional<float> retreat_energy;
  // Rush an enemy whose energy is below this.
  std::optional<float> rush_enemy_energy;
  // Don't bomb an enemy closer than this (tiles).
  std::optional<float> bomb_min_range;
  // Dodge when incoming damage is at least this fraction of max energy.
  std::optional<float> dodge_threshold;
  // Repel (or warp out) when incoming damage is at least this times current energy.
  std::optional<float> repel_trigger;
  // Multifire on when the enemy is further than this (tiles).
  std::optional<float> multifire_range;
  // Antiwarp on above this fraction of max energy.
  std::optional<float> antiwarp_energy;
};

// Stands in for a branch a preset leaves out: always fails, so the selector it's in moves on as if it weren't there.
struct FailureNode : public behavior::BehaviorNode {
  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override { return behavior::ExecuteResult::Failure; }
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

}  // namespace svs
}  // namespace zero
