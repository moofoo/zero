#pragma once

#include <zero/BotController.h>
#include <zero/RegionRegistry.h>
#include <zero/ZeroBot.h>
#include <zero/behavior/BehaviorTree.h>
#include <zero/game/Game.h>

namespace zero {
namespace svs {

// How far the ship has been upgraded by greens, from 0 (initial values) to 1 (every upgradable stat at maximum).
inline float GetShipUpgradeProgress(Game& game, const Player& self) {
  const ShipSettings& settings = game.connection.settings.ShipSettings[self.ship];
  const Ship& ship = game.ship_controller.ship;

  struct Stat {
    float current, initial, maximum;
  };
  const Stat stats[] = {
      {(float)ship.guns, (float)settings.InitialGuns, (float)settings.MaxGuns},
      {(float)ship.bombs, (float)settings.InitialBombs, (float)settings.MaxBombs},
      {(float)ship.energy, (float)settings.InitialEnergy, (float)settings.MaximumEnergy},
      {(float)ship.recharge, (float)settings.InitialRecharge, (float)settings.MaximumRecharge},
      {(float)ship.thrust, (float)settings.InitialThrust, (float)settings.MaximumThrust},
      {(float)ship.speed, (float)settings.InitialSpeed, (float)settings.MaximumSpeed},
      {(float)ship.rotation, (float)settings.InitialRotation, (float)settings.MaximumRotation},
  };

  float total = 0.0f;
  int count = 0;
  for (const Stat& stat : stats) {
    if (stat.maximum <= stat.initial) continue;  // Not upgradable in this zone.

    float progress = (stat.current - stat.initial) / (stat.maximum - stat.initial);
    total += progress < 0.0f ? 0.0f : (progress > 1.0f ? 1.0f : progress);
    ++count;
  }

  return count > 0 ? total / count : 1.0f;
}

// Finds the nearest green. Two optional blackboard values make it pickier, so a bot stops crossing the map for
// prizes it no longer needs:
//   green_upgrade_target (float, 0-1): while the ship's upgrade progress is below this, any green counts.
//   green_max_distance (float, tiles): otherwise, only greens within this distance count.
// With neither set, every green on the map counts.
struct FindNearestGreenNode : public behavior::BehaviorNode {
  FindNearestGreenNode(const char* output_key) : output_key(output_key) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override {
    Player* self = ctx.bot->game->player_manager.GetSelf();
    if (!self || self->ship >= 8) return behavior::ExecuteResult::Failure;

    float max_distance = 1024.0f;

    auto opt_max_distance = ctx.blackboard.Value<float>("green_max_distance");
    if (opt_max_distance) {
      auto opt_upgrade_target = ctx.blackboard.Value<float>("green_upgrade_target");
      bool needs_upgrades = opt_upgrade_target && GetShipUpgradeProgress(*ctx.bot->game, *self) < *opt_upgrade_target;

      if (!needs_upgrades) {
        max_distance = *opt_max_distance;
      }
    }

    PrizeGreen* nearest_green = nullptr;
    float nearest_distance_sq = max_distance * max_distance;

    for (size_t i = 0; i < ctx.bot->game->green_count; ++i) {
      PrizeGreen* green = ctx.bot->game->greens + i;

      float dist_sq = green->position.DistanceSq(self->position);
      if (dist_sq < nearest_distance_sq) {
        nearest_distance_sq = dist_sq;
        nearest_green = green;
      }
    }

    if (!nearest_green) return behavior::ExecuteResult::Failure;

    ctx.blackboard.Set(output_key, nearest_green->position);

    return behavior::ExecuteResult::Success;
  }

  const char* output_key = nullptr;
};

}  // namespace svs
}  // namespace zero
