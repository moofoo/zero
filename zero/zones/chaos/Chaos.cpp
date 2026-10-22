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
#include <zero/zones/chaos/EnemySight.h>
#include <zero/zones/chaos/EnergyEstimate.h>
#include <zero/zones/chaos/Opportunities.h>
#include <zero/zones/chaos/PlayerHistory.h>
#include <zero/zones/chaos/TargetLock.h>
#include <zero/zones/chaos/WeaponNotice.h>

#include <algorithm>
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
// How likely a bot with no Awareness is to notice a shot or hit on its screen, for its estimate of enemy energy.
constexpr float kDefaultShotAwareness = 0.7f;
// The bot's screen, in pixels: what it can see of weapons, shots and hits.
constexpr int kDefaultViewWidth = 1920;
constexpr int kDefaultViewHeight = 1080;
// How long a player takes to notice an enemy that appears only on their radar (seconds), by Awareness: a veteran
// watches the radar, a novice mostly the screen.
struct AwarenessCurve {
  float At(float awareness) const {
    if (awareness <= 0.55f) return 3.0f + (0.55f - awareness) / 0.55f;
    if (awareness <= 0.75f) return 3.0f + (1.5f - 3.0f) * (awareness - 0.55f) / 0.2f;
    return 1.5f + (0.5f - 1.5f) * std::min(1.0f, (awareness - 0.75f) / 0.25f);
  }
};
constexpr AwarenessCurve kRadarNotice;
// A player reacts to a weapon they've noticed this long after, plus their perception delay (ReactionMs): deciding
// which way to go.
constexpr float kDodgeDecisionMs = 150.0f;
// Milliseconds in a game tick.
constexpr float kMsPerTick = 10.0f;
constexpr float kTicksPerSecond = 1000.0f / kMsPerTick;
// Target stickiness: a bot lets go of the enemy it's fighting beyond this distance (tiles), or after this long with
// no damage between them (seconds).
constexpr float kDefaultTargetLeashDistance = 120.0f;
constexpr float kDefaultTargetLockSeconds = 10.0f;
// How far bots hover from the enemy (tiles) without a LeashDistance.
constexpr float kDefaultLeashDistance = 35.0f;

// What Judgement sets, from 0 (a novice: impatient, noisy, slow to decide, overrating its shots and misjudging
// damage) to 1 (a veteran), in a straight line between the two.
struct JudgementRange {
  float at_0;
  float at_1;
  float At(float judgement) const { return at_0 + (at_1 - at_0) * judgement; }
};
constexpr JudgementRange kPatience = {-0.08f, 0.08f};
constexpr JudgementRange kJudgementNoise = {0.15f, 0.03f};
constexpr JudgementRange kDecisionMs = {400.0f, 100.0f};
constexpr JudgementRange kShotJudgement = {0.3f, 0.95f};
constexpr JudgementRange kDamageJudgement = {0.4f, 0.05f};
// From this Judgement on, a pursued bot lays a mine only when the pursuer is on its trail.
constexpr float kMineOnTrailJudgement = 0.5f;
// From this Judgement up, a bot backing off turns on a pursuer that's catching it (novices keep running).
constexpr float kTurnOnPursuerJudgement = 0.5f;

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
// (every bot's). A skill level can so override any style key, e.g. a veteran's own RetreatDistance.
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
    settings.retreat_distance = lookup.GetFloat("RetreatDistance");
    settings.weak_enemy_energy = lookup.GetFloat("WeakEnemyEnergy");
    settings.bomb_min_range = lookup.GetFloat("BombMinRange");
    settings.dodge_threshold = lookup.GetFloat("DodgeThreshold");
    settings.repel_trigger = lookup.GetFloat("RepelTrigger");
    settings.antiwarp_energy = lookup.GetFloat("AntiwarpEnergy");
    settings.bomb_overlap_damage = lookup.GetFloat("BombOverlapDamage");
    settings.shot_spread_range = lookup.GetFloat("ShotSpreadRange");
    settings.bomb_appetite = lookup.GetFloat("BombAppetite").value_or(settings.bomb_appetite);
    auto engage = lookup.Get("Engage");
    settings.engage_close = engage && std::string_view(*engage) == "close";
    settings.reaction_ticks = (int)(skill_float("ReactionMs").value_or(0.0f) / kMsPerTick);
    settings.aim_error = skill_float("AimError").value_or(0.0f);
    settings.lead_skill = skill_float("LeadSkill").value_or(1.0f);
    settings.dodge_error = skill_float("DodgeError").value_or(0.0f);
    settings.piloting = std::clamp(skill_float("Piloting").value_or(1.0f), 0.0f, 1.0f);
    settings.cover_fire = lookup.GetFloat("CoverFire").value_or(1.0f) > 0.5f;
    settings.mine_cover = lookup.GetFloat("MineCover").value_or(1.0f) > 0.5f;
    settings.mine_cover_range = lookup.GetFloat("MineCoverRange").value_or(settings.mine_cover_range);
    settings.accept_confidence = lookup.GetFloat("AcceptConfidence").value_or(settings.accept_confidence);
    auto navigation = lookup.Get("Navigation");
    if (navigation) {
      std::string_view style(*navigation);
      settings.navigation = style == "correction" ? 1 : style == "pursuit" ? 2 : style == "cornering" ? 3 : 0;
    }
    settings.look_ahead = lookup.GetFloat("LookAhead").value_or(settings.look_ahead);
    settings.corner_speed = lookup.GetFloat("CornerSpeed").value_or(settings.corner_speed);
    if (engage && !settings.engage_close && std::string_view(*engage) != "range") {
      Log(LogLevel::Warning, "Engage should be range or close, not %s: using range", *engage);
    }
    Log(LogLevel::Info, "Engage: %s", settings.engage_close ? "close" : "range");

    // Judgement (0-1): how well the bot judges a fight. Without it, the bot judges soundly and without patience.
    DecisionSettings decision_settings;
    auto judgement = skill_float("Judgement");
    if (judgement) {
      float j = std::clamp(*judgement, 0.0f, 1.0f);
      settings.shot_judgement = kShotJudgement.At(j);
      settings.damage_judgement = kDamageJudgement.At(j);
      settings.judgement_noise = kJudgementNoise.At(j);
      settings.mine_on_trail = j >= kMineOnTrailJudgement;
      settings.turns_on_pursuers = j >= kTurnOnPursuerJudgement;
      decision_settings.patience = kPatience.At(j);
      decision_settings.judgement_noise = settings.judgement_noise;
      decision_settings.decision_ticks = (u32)(kDecisionMs.At(j) / kMsPerTick);
      u32 decision_ms = (u32)(decision_settings.decision_ticks * kMsPerTick);
      Log(LogLevel::Info, "Judgement %.2f: patience %+.2f, noise %.2f, deciding every %u ms", j,
          decision_settings.patience, decision_settings.judgement_noise, decision_ms);
    }

    // What the bot can see: weapons, shots and hits on its screen only. The screen is the player's window, the same
    // for every bot, so it's a [Chaos] setting rather than a preset or ship key. Awareness (0-1, skill) is how much of
    // it the bot takes in: the chance of noticing each incoming weapon, for dodging and repelling, and each shot or
    // hit, for its estimate of enemy energy.
    int view_width = config.GetInt(to_string(Zone::Chaos), "ViewWidth").value_or(kDefaultViewWidth);
    int view_height = config.GetInt(to_string(Zone::Chaos), "ViewHeight").value_or(kDefaultViewHeight);
    Vector2f view_extent(view_width / 16.0f / 2.0f, view_height / 16.0f / 2.0f);
    ctx.blackboard.Set("view_extent", view_extent);

    // Weapons count only once noticed, as a player notices them (WeaponNotice), and a dodge reaction time after.
    // Without a skill level, every weapon on screen counts at once.
    auto awareness = skill_float("Awareness");
    if (awareness) awareness = std::clamp(*awareness, 0.0f, 1.0f);
    if (!skill_group.empty()) {
      u32 dodge_reaction = (u32)(kDodgeDecisionMs / kMsPerTick) + (u32)settings.reaction_ticks;
      weapon_notice = std::make_unique<WeaponNotice>(*ctx.bot, awareness.value_or(1.0f), view_extent, dodge_reaction);
      ctx.blackboard.Set<behavior::WeaponNoticer*>("weapon_noticer", weapon_notice.get());
      Log(LogLevel::Info, "Noticing weapons at awareness %.2f, reacting %u ms after", awareness.value_or(1.0f),
          (u32)(dodge_reaction * kMsPerTick));
    }

    // Enemies on the bot's screen are seen at once; one only on its radar once it notices the dot, after a delay that
    // Awareness sets (without a skill level, at once).
    float radar_notice_seconds = awareness ? kRadarNotice.At(*awareness) : 0.0f;
    sight = std::make_unique<EnemySight>(*ctx.bot, view_extent, (u32)(radar_notice_seconds * kTicksPerSecond));
    Log(LogLevel::Info, "Noticing enemies on radar after %.1f s", radar_notice_seconds);

    float shot_awareness = awareness.value_or(kDefaultShotAwareness);
    estimate = std::make_unique<EnergyEstimate>(*ctx.bot, shot_awareness, view_extent);
    history = std::make_unique<PlayerHistory>(*ctx.bot);
    float target_leash = lookup.GetFloat("TargetLeashDistance").value_or(kDefaultTargetLeashDistance);
    float target_lock_seconds = lookup.GetFloat("TargetLockSeconds").value_or(kDefaultTargetLockSeconds);
    // The commit decision: risk appetite from the style keys, judgement from Judgement above.
    decision_settings.commit_confidence =
        lookup.GetFloat("CommitConfidence").value_or(decision_settings.commit_confidence);
    decision_settings.disengage_confidence =
        lookup.GetFloat("DisengageConfidence").value_or(decision_settings.disengage_confidence);
    decision_settings.commit_energy = lookup.GetFloat("CommitEnergy").value_or(decision_settings.commit_energy);
    decision_settings.lead_skill = settings.lead_skill;
    auto leash = lookup.GetFloat("LeashDistance");
    decision_settings.retreat_distance = settings.retreat_distance.value_or(leash.value_or(kDefaultLeashDistance));
    // Spotting opponents' mistakes: as likely as the bot's Judgement (without one, always), noticing their bombs as it
    // notices any weapon.
    opportunities = std::make_unique<Opportunities>(*ctx.bot, judgement ? std::clamp(*judgement, 0.0f, 1.0f) : 1.0f,
                                                    awareness.value_or(1.0f), view_extent);
    decision = std::make_unique<CommitDecision>(*ctx.bot, *estimate, *opportunities, *sight, weapon_notice.get(),
                                                decision_settings);
    Log(LogLevel::Info, "Commit decision: commit at %.2f, disengage below %.2f", decision_settings.commit_confidence,
        decision_settings.disengage_confidence);

    target_lock = std::make_unique<TargetLock>(*ctx.bot, target_leash, (u32)(target_lock_seconds * kTicksPerSecond),
                                               *estimate, view_extent, settings.weak_enemy_energy, *opportunities,
                                               *sight);
    Log(LogLevel::Info, "View %dx%d, awareness of shots and hits %.2f", view_width, view_height, shot_awareness);

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
          .Composite(CreateCombatTree(settings, {*estimate, *history, *target_lock, *decision, *opportunities,
                                                  weapon_notice.get(), *sight}))
          .End();
    // clang-format on

    return builder.Build();
  }

  CombatSettings settings;
  std::unique_ptr<EnergyEstimate> estimate;
  std::unique_ptr<PlayerHistory> history;
  std::unique_ptr<WeaponNotice> weapon_notice;
  std::unique_ptr<EnemySight> sight;
  std::unique_ptr<Opportunities> opportunities;
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
