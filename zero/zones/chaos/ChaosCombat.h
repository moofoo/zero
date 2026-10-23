#pragma once

#include <zero/behavior/BehaviorTree.h>
#include <zero/zones/chaos/Places.h>

#include <memory>
#include <optional>
#include <vector>

namespace zero {
namespace chaos {

struct CommitDecision;
struct EnemySight;
struct EnergyEstimate;
struct Opportunities;
struct PlayerHistory;
struct TargetLock;
struct WeaponNotice;

// How a Chaos bot fights. Every ship flies the same combat tree; these settings decide its style. Chaos reads each
// style key from the bot's skill level ([Skill.<name>], which may override style), then its aggression preset
// ([Aggression.<name>], chosen with --aggression), then its ship's section ([Ship.<name>], e.g. the Terrier's closer
// style), then [Chaos], and a value set in none keeps the built-in number. A trait whose value is unset (dodge, the
// weak-enemy switch, bomb overlap, shot spread) is left out.
struct CombatSettings {
  // Places on the map to roam between (ticket 09); without any, the bot patrols the Waypoints route.
  std::vector<Place> places;
  // Back off to this distance from the enemy (tiles); without it, to the leash distance.
  std::optional<float> retreat_distance;
  // Switch to the weakest enemy on screen whose estimated energy is below this (the target lock), and stay on it.
  std::optional<float> weak_enemy_energy;
  // Don't bomb an enemy closer than this (tiles).
  std::optional<float> bomb_min_range;
  // Dodge when incoming damage is at least this fraction of max energy.
  std::optional<float> dodge_threshold;
  // Repel (or warp out) when incoming damage is at least this times current energy.
  std::optional<float> repel_trigger;
  // Ship abilities: antiwarp on above this fraction of max energy.
  std::optional<float> antiwarp_energy;
  // Bomb only when at least this much of our own bullet damage is already heading at the target, so the bomb
  // lands among bullets and is harder to dodge.
  std::optional<float> bomb_overlap_damage;
  // Spread shots around targets further than this (tiles).
  std::optional<float> shot_spread_range;
  // Close in on the target, rather than hovering at the leash distance.
  bool engage_close = false;
  // How readily the bot bombs: a multiplier on a bomb's judged worth in the shot choice (1: as judged).
  float bomb_appetite = 1.0f;

  // Skill (a [Skill.<name>] section, chosen with --skill). The defaults are a bot with no skill level: perfect aim,
  // instant reactions, sound judgement.
  // Aim at the target as seen this many ticks ago (weapons get the same delay through the view).
  int reaction_ticks = 0;
  // Aim drifts off target by about this many degrees.
  float aim_error = 0.0f;
  // How much of the target's movement the bot leads its shots by (1: a full intercept).
  float lead_skill = 1.0f;
  // How often a sideways dodge goes the wrong way.
  float dodge_error = 0.0f;
  // How well the bot times its thrust in a fight, 0-1: 0 thrusts whenever it wants to move, however its ship faces;
  // 1 thrusts only when its heading helps, by the commit decision's mode (ticket 19).
  float piloting = 1.0f;
  // From Judgement: how well the bot judges a shot's chance of hitting (1: accurately; lower overestimates it, so the
  // bot sprays), how far off its judgement of a hit's damage is, and of its chance of winning.
  float shot_judgement = 1.0f;
  float damage_judgement = 0.0f;
  float judgement_noise = 0.0f;
  // From Judgement: while pursued, lay a mine only when the pursuer is on the bot's trail.
  bool mine_on_trail = false;
  // From Judgement: backing off and being chased down, turn on the pursuer (ticket 30). Sound judgement does.
  bool turns_on_pursuers = true;
  // From Judgement: switch cloak and stealth off under pressure, for the regeneration (ticket 29). Sound judgement
  // does; a novice leaves them on in a fight.
  bool manages_cloak = true;
  // From Judgement: a cloaked Spider stalks prey and strikes (ticket 29); veterans only.
  bool stalks = true;

  // While backing off and pursued: fire cover shots back at the pursuer however slow, and lay mines in its path when
  // it's within this range (tiles) (style).
  bool cover_fire = true;
  bool mine_cover = true;
  float mine_cover_range = 25.0f;
  // Accepting a hit while committing or pressuring: when the win chance after taking it is at least this (style; 1:
  // never).
  float accept_confidence = 0.55f;

  // Following the patrol route: 0 classic (face each waypoint), 1 correction, 2 pursuit, 3 cornering (see RouteNode);
  // how far ahead along the route pursuit steers (tiles), and the speed cornering allows at a right angle (tiles/s).
  int navigation = 0;
  float look_ahead = 12.0f;
  float corner_speed = 8.0f;
};

// The tree every Chaos bot flies: enter its ship, then fight the nearest enemy, or else chase greens, or else
// patrol the waypoints.
// What the bot knows and remembers, which must outlive the tree: enemy energy only from the estimate, with a reaction
// time the target's movement from the history, whom it's fighting from the target lock, and its opponents' mistakes
// it has spotted.
struct CombatSenses {
  EnergyEstimate& estimate;
  PlayerHistory& history;
  TargetLock& target_lock;
  CommitDecision& decision;
  Opportunities& opportunities;
  // Which enemy weapons the bot has noticed; null for a bot without a skill level, which sees every weapon at once.
  WeaponNotice* weapon_notice;
  // Which enemies the bot can see, on its screen or radar.
  EnemySight& sight;
};

std::unique_ptr<behavior::BehaviorNode> CreateCombatTree(const CombatSettings& settings, CombatSenses senses);

}  // namespace chaos
}  // namespace zero
