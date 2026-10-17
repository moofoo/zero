#include <stdlib.h>
#include <zero/BotController.h>
#include <zero/ZeroBot.h>
#include <zero/behavior/BehaviorBuilder.h>
#include <zero/behavior/nodes/BlackboardNode.h>
#include <zero/behavior/nodes/MathNode.h>
#include <zero/behavior/nodes/PlayerNode.h>
#include <zero/behavior/nodes/TimerNode.h>
#include <zero/game/Logger.h>
#include <zero/zones/ZoneController.h>
#include <zero/zones/svs/TerrierBehavior.h>
#include <zero/zones/svs/WarbirdBehavior.h>

#include <string>
#include <string_view>
#include <vector>

namespace zero {
namespace chaos {

// Parses waypoints written as "{ x, y }; { x, y }; ...".
static std::vector<Vector2f> ParseWaypoints(std::string_view text) {
  std::vector<Vector2f> waypoints;

  size_t open = text.find('{');
  while (open != std::string_view::npos) {
    size_t close = text.find('}', open);
    if (close == std::string_view::npos) break;

    std::string pair(text.substr(open + 1, close - open - 1));
    size_t comma = pair.find(',');
    if (comma != std::string::npos) {
      float x = (float)strtol(pair.c_str(), nullptr, 10);
      float y = (float)strtol(pair.c_str() + comma + 1, nullptr, 10);
      waypoints.emplace_back(x, y);
    }

    open = text.find('{', close);
  }

  return waypoints;
}

constexpr float kDefaultGreenUpgradeTarget = 0.75f;
constexpr float kDefaultGreenMaxDistance = 25.0f;

// Wraps an SVS behavior for Chaos: idle waypoints come from the [Chaos] Waypoints config instead of the SVS
// map's, greens are chased only while needed or nearby, and the bot first switches to the [Chaos] Freq config
// value (set by --freq) when there is one.
template <typename SvsBehavior>
struct ChaosBehavior : public SvsBehavior {
  void OnInitialize(behavior::ExecuteContext& ctx) override {
    SvsBehavior::OnInitialize(ctx);

    // Without configured waypoints the list is empty, so the bot holds position when there's nothing to fight
    // rather than flying at SVS map coordinates that land in Chaos walls.
    std::vector<Vector2f> waypoints;
    auto opt_waypoints = ctx.bot->config->GetString(to_string(Zone::Chaos), "Waypoints");
    if (opt_waypoints) {
      waypoints = ParseWaypoints(*opt_waypoints);
    }

    ctx.blackboard.Set("waypoints", waypoints);

    // Chase greens like a player: anywhere while the ship still needs upgrades, then only nearby ones.
    ctx.blackboard.Set("green_upgrade_target", ReadFloat(ctx, "GreenUpgradeTarget", kDefaultGreenUpgradeTarget));
    ctx.blackboard.Set("green_max_distance", ReadFloat(ctx, "GreenMaxDistance", kDefaultGreenMaxDistance));
  }

  static float ReadFloat(behavior::ExecuteContext& ctx, const char* key, float default_value) {
    auto opt_value = ctx.bot->config->GetString(to_string(Zone::Chaos), key);
    return opt_value ? (float)atof(*opt_value) : default_value;
  }

  std::unique_ptr<behavior::BehaviorNode> CreateTree(behavior::ExecuteContext& ctx) override {
    using namespace behavior;

    BehaviorBuilder builder;

    // clang-format off
    builder
      .Selector()
          .Sequence() // Switch to the requested frequency when one is set.
              .Child<ReadConfigIntNode<u16>>("Freq", "request_freq")
              .Child<PlayerFrequencyQueryNode>("self_freq")
              .InvertChild<EqualityNode<u16>>("self_freq", "request_freq")
              .Child<TimerExpiredNode>("next_freq_change_tick")
              .Child<TimerSetNode>("next_freq_change_tick", 300)
              .Child<PlayerChangeFrequencyNode>("request_freq")
              .End()
          .Composite(SvsBehavior::CreateTree(ctx))
          .End();
    // clang-format on

    return builder.Build();
  }
};

struct ChaosController : ZoneController {
  bool IsZone(Zone zone) override { return zone == Zone::Chaos; }

  void CreateBehaviors(const char* arena_name) override;
};

static ChaosController controller;

void ChaosController::CreateBehaviors(const char* arena_name) {
  Log(LogLevel::Info, "Registering Chaos behaviors.");

  bot->bot_controller->energy_tracker.estimate_type = EnergyHeuristicType::Average;

  // --freq overrides the Freq config value.
  std::string_view freq_override = bot->args->GetValue({"freq"});
  if (!freq_override.empty()) {
    bot->config->GetOrCreateGroup(to_string(Zone::Chaos)).map["Freq"] = std::string(freq_override);
  }

  auto& repo = bot->bot_controller->behaviors;

  repo.Add("warbird", std::make_unique<ChaosBehavior<svs::WarbirdBehavior>>());
  repo.Add("terrier", std::make_unique<ChaosBehavior<svs::TerrierBehavior>>());

  SetBehavior("warbird");
}

}  // namespace chaos
}  // namespace zero
