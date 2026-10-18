#pragma once

#include <zero/behavior/BehaviorTree.h>

#include <memory>
#include <optional>

namespace zero {
namespace chaos {

struct EnergyEstimate;
struct PlayerHistory;

// How a Chaos bot fights. Every ship flies the same combat tree; these settings decide its style. Chaos reads them
// from the bot's aggression preset ([Aggression.<name>], chosen with --aggression), then its ship's section
// ([Ship.<name>], e.g. the Terrier's closer, more aggressive style), and a value set in neither keeps the built-in
// number. A trait whose value is unset (dodge, rush, retreat, bomb overlap, shot spread) is left out.
struct CombatSettings {
  // Fire bombs / bullets only above this fraction of max energy.
  std::optional<float> bomb_energy;
  std::optional<float> bullet_energy;
  // Back off below this fraction of max energy: to the retreat distance, or without one to the leash distance.
  std::optional<float> retreat_energy;
  std::optional<float> retreat_distance;
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
  // Bomb only when at least this much of our own bullet damage is already heading at the target, so the bomb
  // lands among bullets and is harder to dodge.
  std::optional<float> bomb_overlap_damage;
  // Spread shots around targets further than this (tiles).
  std::optional<float> shot_spread_range;
  // Close in on the target, rather than hovering at the leash distance.
  bool engage_close = false;

  // Skill (a [Skill.<name>] section, chosen with --skill). The defaults are today's bots: perfect aim, instant
  // reactions.
  // Aim at the target as seen this many ticks ago (weapons get the same delay through the view).
  int reaction_ticks = 0;
  // Aim drifts off target by about this many degrees.
  float aim_error = 0.0f;
  // Scale of the box a shot's path must cross to be fired; larger takes more low-percentage shots.
  float hit_box_scale = 1.0f;
  // Hold shots slower than this along the line to the target (tiles/s), except while backing off.
  float min_shot_speed = 0.0f;
};

// The tree every Chaos bot flies: enter its ship, then fight the nearest enemy, or else chase greens, or else
// patrol the waypoints. Enemy energy comes only from the estimate, and with a reaction time the target's movement
// from the history; both must outlive the tree.
std::unique_ptr<behavior::BehaviorNode> CreateCombatTree(const CombatSettings& settings, EnergyEstimate& estimate,
                                                         PlayerHistory& history);

}  // namespace chaos
}  // namespace zero
