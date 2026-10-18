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

#include <optional>
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

// A float from the config, if the key is there.
static std::optional<float> ReadOptionalFloat(Config& config, const std::string& group, const char* key) {
  auto opt_value = config.GetString(group, key);
  if (!opt_value) return {};
  return (float)atof(*opt_value);
}

// Wraps an SVS behavior for Chaos: idle waypoints come from the [Chaos] Waypoints config instead of the SVS
// map's, greens are chased only while needed or nearby, the bot first switches to the [Chaos] Freq config
// value (set by --freq) when there is one, and an aggression preset (--aggression <name>, read from the
// [Aggression.<name>] config section) can change the tree's tactical thresholds.
template <typename SvsBehavior>
struct ChaosBehavior : public SvsBehavior {
  void OnInitialize(behavior::ExecuteContext& ctx) override {
    SvsBehavior::OnInitialize(ctx);

    // A section with no keys isn't kept by the config loader, so a preset that changes nothing (standard) and a
    // missing one look the same: the tree's own thresholds. chaos.py checks the names before starting bots.
    std::string preset(ctx.bot->args->GetValue({"aggression"}));
    std::string group = "Aggression." + preset;
    Config& config = *ctx.bot->config;
    if (!preset.empty()) {
      svs::AggressionSettings& aggression = this->aggression;
      aggression.bomb_energy = ReadOptionalFloat(config, group, "BombEnergy");
      aggression.bullet_energy = ReadOptionalFloat(config, group, "BulletEnergy");
      aggression.retreat_energy = ReadOptionalFloat(config, group, "RetreatEnergy");
      aggression.rush_enemy_energy = ReadOptionalFloat(config, group, "RushEnemyEnergy");
      aggression.bomb_min_range = ReadOptionalFloat(config, group, "BombMinRange");
      aggression.dodge_threshold = ReadOptionalFloat(config, group, "DodgeThreshold");
      aggression.repel_trigger = ReadOptionalFloat(config, group, "RepelTrigger");
      aggression.multifire_range = ReadOptionalFloat(config, group, "MultifireRange");
      aggression.antiwarp_energy = ReadOptionalFloat(config, group, "AntiwarpEnergy");
      Log(LogLevel::Info, "Aggression preset: %s", preset.c_str());

      auto leash_distance = ReadOptionalFloat(config, group, "LeashDistance");
      if (leash_distance) ctx.blackboard.Set("leash_distance", *leash_distance);
    }

    // Without configured waypoints the list is empty, so the bot holds position when there's nothing to fight
    // rather than flying at SVS map coordinates that land in Chaos walls.
    std::vector<Vector2f> waypoints;
    auto opt_waypoints = ctx.bot->config->GetString(to_string(Zone::Chaos), "Waypoints");
    if (opt_waypoints) {
      waypoints = ParseWaypoints(*opt_waypoints);
    }

    ctx.blackboard.Set("waypoints", waypoints);

    // Chase greens like a player: anywhere while the ship still needs upgrades, then only nearby ones.
    // The preset's values win over [Chaos].
    float green_upgrade_target =
        ReadOptionalFloat(config, to_string(Zone::Chaos), "GreenUpgradeTarget").value_or(kDefaultGreenUpgradeTarget);
    float green_max_distance =
        ReadOptionalFloat(config, to_string(Zone::Chaos), "GreenMaxDistance").value_or(kDefaultGreenMaxDistance);
    if (!preset.empty()) {
      green_upgrade_target = ReadOptionalFloat(config, group, "GreenUpgradeTarget").value_or(green_upgrade_target);
      green_max_distance = ReadOptionalFloat(config, group, "GreenMaxDistance").value_or(green_max_distance);
    }
    ctx.blackboard.Set("green_upgrade_target", green_upgrade_target);
    ctx.blackboard.Set("green_max_distance", green_max_distance);
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
