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
#include <zero/zones/chaos/ChaosCombat.h>

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

// The ship names [Ship.<name>] config sections use, by ship number (0 is the Warbird).
static const char* const kShipNames[] = {"Warbird", "Javelin", "Spider", "Leviathan",
                                         "Terrier", "Weasel",  "Lancaster", "Shark"};

// The ship the bot will fly (0-7) from --ship, else the RequestShip config value; -1 if neither names a ship.
static int GetRequestedShip(ZeroBot& bot) {
  std::string_view ship_override = bot.args->GetValue({"ship"});
  if (!ship_override.empty()) {
    int ship = ship_override[0] - '1';
    return ship >= 0 && ship < (int)ZERO_ARRAY_SIZE(kShipNames) ? ship : -1;
  }

  const char* group_lookups[] = {to_string(Zone::Chaos), "General"};
  auto request_ship = bot.config->GetInt(group_lookups, ZERO_ARRAY_SIZE(group_lookups), "RequestShip");
  int ship = request_ship ? *request_ship - 1 : -1;
  return ship >= 0 && ship < (int)ZERO_ARRAY_SIZE(kShipNames) ? ship : -1;
}

// Looks a setting up in the bot's aggression preset, then its ship's section.
struct SettingLookup {
  Config& config;
  std::string preset_group;
  std::string ship_group;

  std::optional<const char*> Get(const char* key) const {
    std::optional<const char*> value;
    if (!preset_group.empty()) value = config.GetString(preset_group, key);
    if (!value && !ship_group.empty()) value = config.GetString(ship_group, key);
    return value;
  }

  std::optional<float> GetFloat(const char* key) const {
    std::optional<float> value;
    if (!preset_group.empty()) value = ReadOptionalFloat(config, preset_group, key);
    if (!value && !ship_group.empty()) value = ReadOptionalFloat(config, ship_group, key);
    return value;
  }
};

// The Chaos bot. Every ship flies the same combat tree (ChaosCombat.h), styled by the bot's aggression preset
// (--aggression <name>, read from [Aggression.<name>]) and its ship's section ([Ship.<name>]). Idle waypoints come
// from the [Chaos] Waypoints config, greens are chased only while needed or nearby, and the bot first switches to
// the [Chaos] Freq config value (set by --freq) when there is one.
struct ChaosBehavior : public behavior::Behavior {
  void OnInitialize(behavior::ExecuteContext& ctx) override {
    // The zone controller sets request_ship from --ship after this.
    ctx.blackboard.Set("request_ship", 0);
    ctx.blackboard.Set("leash_distance", 35.0f);

    // A section with no keys isn't kept by the config loader, so a preset that changes nothing (standard) and a
    // missing one look the same. chaos.py checks the names before starting bots.
    Config& config = *ctx.bot->config;
    std::string preset(ctx.bot->args->GetValue({"aggression"}));
    int ship = GetRequestedShip(*ctx.bot);
    SettingLookup lookup{config, preset.empty() ? std::string() : "Aggression." + preset,
                         ship >= 0 && ship < (int)ZERO_ARRAY_SIZE(kShipNames) ? std::string("Ship.") + kShipNames[ship] : std::string()};
    if (!preset.empty()) Log(LogLevel::Info, "Aggression preset: %s", preset.c_str());

    settings = CombatSettings();
    settings.bomb_energy = lookup.GetFloat("BombEnergy");
    settings.bullet_energy = lookup.GetFloat("BulletEnergy");
    settings.retreat_energy = lookup.GetFloat("RetreatEnergy");
    settings.rush_enemy_energy = lookup.GetFloat("RushEnemyEnergy");
    settings.bomb_min_range = lookup.GetFloat("BombMinRange");
    settings.dodge_threshold = lookup.GetFloat("DodgeThreshold");
    settings.repel_trigger = lookup.GetFloat("RepelTrigger");
    settings.multifire_range = lookup.GetFloat("MultifireRange");
    settings.antiwarp_energy = lookup.GetFloat("AntiwarpEnergy");
    settings.bomb_overlap_damage = lookup.GetFloat("BombOverlapDamage");
    settings.shot_spread_range = lookup.GetFloat("ShotSpreadRange");
    auto engage = lookup.Get("Engage");
    settings.engage_close = engage && std::string_view(*engage) == "close";
    if (engage && !settings.engage_close && std::string_view(*engage) != "range") {
      Log(LogLevel::Warning, "Engage should be range or close, not %s: using range", *engage);
    }
    Log(LogLevel::Info, "Engage: %s", settings.engage_close ? "close" : "range");

    auto leash_distance = lookup.GetFloat("LeashDistance");
    if (leash_distance) ctx.blackboard.Set("leash_distance", *leash_distance);

    // Without configured waypoints the list is empty, so the bot holds position when there's nothing to fight
    // rather than flying at SVS map coordinates that land in Chaos walls.
    std::vector<Vector2f> waypoints;
    auto opt_waypoints = ctx.bot->config->GetString(to_string(Zone::Chaos), "Waypoints");
    if (opt_waypoints) {
      waypoints = ParseWaypoints(*opt_waypoints);
    }

    ctx.blackboard.Set("waypoints", waypoints);

    // Chase greens like a player: anywhere while the ship still needs upgrades, then only nearby ones.
    // The preset's or ship's values win over [Chaos].
    float green_upgrade_target =
        ReadOptionalFloat(config, to_string(Zone::Chaos), "GreenUpgradeTarget").value_or(kDefaultGreenUpgradeTarget);
    float green_max_distance =
        ReadOptionalFloat(config, to_string(Zone::Chaos), "GreenMaxDistance").value_or(kDefaultGreenMaxDistance);
    green_upgrade_target = lookup.GetFloat("GreenUpgradeTarget").value_or(green_upgrade_target);
    green_max_distance = lookup.GetFloat("GreenMaxDistance").value_or(green_max_distance);
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
          .Composite(CreateCombatTree(settings))
          .End();
    // clang-format on

    return builder.Build();
  }

  CombatSettings settings;
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

  repo.Add("chaos", std::make_unique<ChaosBehavior>());

  SetBehavior("chaos");
}

}  // namespace chaos
}  // namespace zero
