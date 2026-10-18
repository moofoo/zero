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
#include <zero/zones/chaos/CommitDecision.h>
#include <zero/zones/chaos/EnergyEstimate.h>
#include <zero/zones/chaos/PlayerHistory.h>
#include <zero/zones/chaos/TargetLock.h>

#include <memory>
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
// How likely the bot is to notice a shot or hit on its screen, for its estimate of enemy energy.
constexpr float kDefaultAttention = 0.7f;
// The bot's screen, in pixels: what it can see of weapons, shots and hits.
constexpr int kDefaultViewWidth = 1920;
constexpr int kDefaultViewHeight = 1080;
// Milliseconds in a game tick.
constexpr float kMsPerTick = 10.0f;
constexpr float kTicksPerSecond = 1000.0f / kMsPerTick;
// Target stickiness: a bot lets go of the enemy it's fighting beyond this distance (tiles), or after this long with
// no damage between them (seconds).
constexpr float kDefaultTargetLeashDistance = 120.0f;
constexpr float kDefaultTargetLockSeconds = 10.0f;
// How far bots hover from the enemy (tiles) without a LeashDistance.
constexpr float kDefaultLeashDistance = 35.0f;

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

// Looks a setting up in the bot's skill level, then its aggression preset, then its ship's section, then [Chaos]
// (every bot's). A skill level can so override any style key, e.g. a veteran's own RetreatEnergy.
struct SettingLookup {
  Config& config;
  std::string skill_group;
  std::string preset_group;
  std::string ship_group;

  std::optional<const char*> Get(const char* key) const {
    std::optional<const char*> value;
    if (!skill_group.empty()) value = config.GetString(skill_group, key);
    if (!value && !preset_group.empty()) value = config.GetString(preset_group, key);
    if (!value && !ship_group.empty()) value = config.GetString(ship_group, key);
    if (!value) value = config.GetString(to_string(Zone::Chaos), key);
    return value;
  }

  std::optional<float> GetFloat(const char* key) const {
    auto value = Get(key);
    if (!value) return {};
    return (float)atof(*value);
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
    ctx.blackboard.Set("leash_distance", kDefaultLeashDistance);

    // A section with no keys isn't kept by the config loader, so a preset that changes nothing (standard) and a
    // missing one look the same. chaos.py checks the names before starting bots.
    Config& config = *ctx.bot->config;
    std::string preset(ctx.bot->args->GetValue({"aggression"}));
    int ship = GetRequestedShip(*ctx.bot);
    std::string ship_group;
    if (ship >= 0 && ship < (int)ZERO_ARRAY_SIZE(kShipNames)) ship_group = std::string("Ship.") + kShipNames[ship];
    // Skill: how well the bot plays. Its skill keys come from its [Skill.<name>] section only; without one, aim and
    // reactions are perfect. The section can also set style keys, which win over the preset's and ship's.
    std::string skill(ctx.bot->args->GetValue({"skill"}));
    std::string skill_group = skill.empty() ? std::string() : "Skill." + skill;
    SettingLookup lookup{config, skill_group, preset.empty() ? std::string() : "Aggression." + preset, ship_group};
    if (!preset.empty()) Log(LogLevel::Info, "Aggression preset: %s", preset.c_str());
    auto skill_float = [&](const char* key) -> std::optional<float> {
      if (skill_group.empty()) return {};
      return ReadOptionalFloat(config, skill_group, key);
    };
    if (!skill.empty()) Log(LogLevel::Info, "Skill: %s", skill.c_str());

    settings = CombatSettings();
    settings.bomb_energy = lookup.GetFloat("BombEnergy");
    settings.bullet_energy = lookup.GetFloat("BulletEnergy");
    settings.retreat_energy = lookup.GetFloat("RetreatEnergy");
    settings.retreat_distance = lookup.GetFloat("RetreatDistance");
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
    settings.reaction_ticks = (int)(skill_float("ReactionMs").value_or(0.0f) / kMsPerTick);
    settings.aim_error = skill_float("AimError").value_or(0.0f);
    settings.hit_box_scale = skill_float("HitBoxScale").value_or(1.0f);
    settings.min_shot_speed = skill_float("MinShotSpeed").value_or(0.0f);
    settings.commit_decision = lookup.GetFloat("CommitDecision").value_or(0.0f) > 0.5f;
    settings.shot_judgement = skill_float("ShotJudgement").value_or(1.0f);
    if (engage && !settings.engage_close && std::string_view(*engage) != "range") {
      Log(LogLevel::Warning, "Engage should be range or close, not %s: using range", *engage);
    }
    Log(LogLevel::Info, "Engage: %s", settings.engage_close ? "close" : "range");

    // What the bot can see: weapons, shots and hits on its screen only. The screen is the player's window, the same
    // for every bot, so it's a [Chaos] setting rather than a preset or ship key; attention is a trait of the bot.
    int view_width = config.GetInt(to_string(Zone::Chaos), "ViewWidth").value_or(kDefaultViewWidth);
    int view_height = config.GetInt(to_string(Zone::Chaos), "ViewHeight").value_or(kDefaultViewHeight);
    Vector2f view_extent(view_width / 16.0f / 2.0f, view_height / 16.0f / 2.0f);
    ctx.blackboard.Set("view_extent", view_extent);

    // Weapons count only once they've existed for the reaction time, and if noticed.
    ctx.blackboard.Set("weapon_reaction_ticks", settings.reaction_ticks);
    ctx.blackboard.Set("weapon_notice", skill_float("NoticeWeapon").value_or(1.0f));

    float attention = lookup.GetFloat("Attention").value_or(kDefaultAttention);
    estimate = std::make_unique<EnergyEstimate>(*ctx.bot, attention, view_extent);
    history = std::make_unique<PlayerHistory>(*ctx.bot);
    float target_leash = lookup.GetFloat("TargetLeashDistance").value_or(kDefaultTargetLeashDistance);
    float target_lock_seconds = lookup.GetFloat("TargetLockSeconds").value_or(kDefaultTargetLockSeconds);
    // The commit decision: risk appetite from the style keys, judgement from the skill keys.
    DecisionSettings decision_settings;
    decision_settings.commit_confidence =
        lookup.GetFloat("CommitConfidence").value_or(decision_settings.commit_confidence);
    decision_settings.disengage_confidence =
        lookup.GetFloat("DisengageConfidence").value_or(decision_settings.disengage_confidence);
    decision_settings.judgement_noise = skill_float("JudgementNoise").value_or(decision_settings.judgement_noise);
    auto decision_ms = skill_float("DecisionMs");
    if (decision_ms) decision_settings.decision_ticks = (u32)(*decision_ms / kMsPerTick);
    auto leash = lookup.GetFloat("LeashDistance");
    decision_settings.retreat_distance = settings.retreat_distance.value_or(leash.value_or(kDefaultLeashDistance));
    decision = std::make_unique<CommitDecision>(*ctx.bot, *estimate, decision_settings);
    if (settings.commit_decision) {
      Log(LogLevel::Info, "Commit decision: commit at %.2f, disengage below %.2f, noise %.2f",
          decision_settings.commit_confidence, decision_settings.disengage_confidence,
          decision_settings.judgement_noise);
    }

    target_lock = std::make_unique<TargetLock>(*ctx.bot, target_leash, (u32)(target_lock_seconds * kTicksPerSecond),
                                               *estimate, view_extent, settings.rush_enemy_energy);
    Log(LogLevel::Info, "View %dx%d, attention %.2f", view_width, view_height, attention);

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
    float green_upgrade_target = lookup.GetFloat("GreenUpgradeTarget").value_or(kDefaultGreenUpgradeTarget);
    float green_max_distance = lookup.GetFloat("GreenMaxDistance").value_or(kDefaultGreenMaxDistance);
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
          .Composite(CreateCombatTree(settings, {*estimate, *history, *target_lock, *decision}))
          .End();
    // clang-format on

    return builder.Build();
  }

  CombatSettings settings;
  std::unique_ptr<EnergyEstimate> estimate;
  std::unique_ptr<PlayerHistory> history;
  std::unique_ptr<TargetLock> target_lock;
  std::unique_ptr<CommitDecision> decision;
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
