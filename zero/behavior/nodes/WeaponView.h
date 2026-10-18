#pragma once

#include <zero/Math.h>
#include <zero/behavior/BehaviorTree.h>
#include <zero/game/WeaponManager.h>

namespace zero {
namespace behavior {

// Which weapons a node may react to. A behavior that should react only to what a player in its seat could see sets
// the "view_extent" blackboard key to half its screen's size, in tiles; without that key every weapon counts.
struct WeaponView {
  WeaponView(ExecuteContext& ctx, Vector2f center) : center(center) {
    auto opt_extent = ctx.blackboard.Value<Vector2f>("view_extent");
    if (opt_extent) {
      limited = true;
      extent = *opt_extent;
    }
  }

  bool Contains(const Weapon& weapon) const {
    if (!limited) return true;

    Vector2f delta = weapon.position - center;
    return fabsf(delta.x) <= extent.x && fabsf(delta.y) <= extent.y;
  }

  Vector2f center;
  Vector2f extent;
  bool limited = false;
};

}  // namespace behavior
}  // namespace zero
