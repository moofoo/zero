#include "EnergyEstimate.h"

#include <math.h>
#include <zero/HeuristicEnergyTracker.h>
#include <zero/ZeroBot.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

// The standard deviation of a value spread evenly over a range of this width.
static float UniformSd(float width) {
  return width / sqrtf(12.0f);
}

// How far off a player's judgement of a noticed shot's cost is, as a fraction of it.
constexpr float kShotCostNoise = 0.1f;
// A belief isn't advanced over more than this many ticks at once (a minute): longer than any recharge.
constexpr s32 kMaxAdvanceTicks = 6000;
// Off screen, the estimate relaxes towards this fraction of the average maximum energy, a typical energy in a fight
// (bots go into exchanges at about 60% on average), with this time constant in ticks.
constexpr float kUnseenTypicalEnergy = 0.6f;
constexpr float kUnseenTicks = 500.0f;

EnergyEstimate::EnergyEstimate(ZeroBot& bot, float attention, Vector2f view_extent)
    : bot(bot), attention(attention), view_extent(view_extent), random(std::random_device{}()) {}

EnergyEstimate::Reading EnergyEstimate::Get(Player& player) {
  // Energy the server sent, as it would show it to a player, while it's fresh.
  if (player.energy > 0 && TICK_DIFF(GetCurrentTick(), player.last_extra_timestamp) < kExtraDataTimeout) {
    return {(float)player.energy, 0.0f};
  }

  Belief& belief = Advance(player);
  return {belief.mean, belief.sd};
}

void EnergyEstimate::Spawn(Player& player) {
  Belief& belief = beliefs[player.id];
  belief = Belief();
  if (player.ship >= 8) return;

  auto& settings = bot.game->connection.settings.ShipSettings[player.ship];
  belief.mean = (settings.InitialEnergy + settings.MaximumEnergy) / 2.0f;
  belief.sd = UniformSd((float)(settings.MaximumEnergy - settings.InitialEnergy));
  belief.tick = GetCurrentTick();
  belief.valid = true;
}

EnergyEstimate::Belief& EnergyEstimate::Advance(Player& player) {
  Belief& belief = beliefs[player.id];
  if (!belief.valid) {
    Spawn(player);
    return belief;
  }

  u32 now = GetCurrentTick();
  s32 ticks = TICK_DIFF(now, belief.tick);
  belief.tick = now;
  if (ticks <= 0 || player.ship >= 8) return belief;
  if (ticks > kMaxAdvanceTicks) ticks = kMaxAdvanceTicks;

  auto& settings = bot.game->connection.settings.ShipSettings[player.ship];
  float max_energy = (settings.InitialEnergy + settings.MaximumEnergy) / 2.0f;

  if (!InView(player.position)) {
    // Unseen, they could be fighting anyone: drift towards a typical energy, anything from empty to full. The
    // result is the mixture of the old belief and that prior.
    float keep = expf(-ticks / kUnseenTicks);
    float typical = max_energy * kUnseenTypicalEnergy;
    float typical_sd = UniformSd(max_energy);
    float mean = keep * belief.mean + (1.0f - keep) * typical;
    float variance = keep * belief.sd * belief.sd + (1.0f - keep) * typical_sd * typical_sd +
                     keep * (1.0f - keep) * (belief.mean - typical) * (belief.mean - typical);
    belief.mean = mean;
    belief.sd = sqrtf(variance);
    return belief;
  }

  // Recharge settings are energy per 10 seconds, so per tick they're divided by 1000.
  float recharge = (settings.InitialRecharge + settings.MaximumRecharge) / 2.0f / 1000.0f;
  float recharge_sd = UniformSd((float)(settings.MaximumRecharge - settings.InitialRecharge)) / 1000.0f;

  belief.mean += recharge * ticks;
  belief.sd += recharge_sd * ticks;

  // Recharged to full: what's left to doubt is how high full is.
  if (belief.mean >= max_energy) {
    belief.mean = max_energy;
    belief.sd = UniformSd((float)(settings.MaximumEnergy - settings.InitialEnergy));
  }

  return belief;
}

bool EnergyEstimate::InView(Vector2f position) {
  Player* self = bot.game->player_manager.GetSelf();
  if (!self || self->ship >= 8) return false;

  Vector2f delta = position - self->position;
  return fabsf(delta.x) <= view_extent.x && fabsf(delta.y) <= view_extent.y;
}

bool EnergyEstimate::Notices(Vector2f position) {
  if (!InView(position)) return false;

  return std::uniform_real_distribution<float>(0.0f, 1.0f)(random) < attention;
}

void EnergyEstimate::Spend(Player& player, float amount, float amount_sd) {
  Belief& belief = Advance(player);

  // For each event noticed, about 1/attention happened (the count of those missed is geometric).
  if (attention > 0.0f && attention < 1.0f) {
    float missed_sd = amount * sqrtf(1.0f - attention) / attention;
    amount /= attention;
    amount_sd = sqrtf(amount_sd * amount_sd + missed_sd * missed_sd);
  }

  belief.mean -= amount;
  if (belief.mean < 0.0f) belief.mean = 0.0f;
  belief.sd = sqrtf(belief.sd * belief.sd + amount_sd * amount_sd);
}

void EnergyEstimate::HandleEvent(const WeaponFireEvent& event) {
  Player& player = event.player;
  if (player.ship >= 8 || player.id == bot.game->player_manager.player_id) return;
  if (!Notices(player.position)) return;

  float cost = (float)GetWeaponFireCost(bot.game->connection.settings.ShipSettings[player.ship], event.data);
  if (cost <= 0.0f) return;

  float judged = cost * std::normal_distribution<float>(1.0f, kShotCostNoise)(random);
  Spend(player, judged > 0.0f ? judged : 0.0f, cost * kShotCostNoise);
}

void EnergyEstimate::HandleEvent(const WeaponHitEvent& event) {
  Weapon& weapon = event.weapon;
  auto& settings = bot.game->connection.settings;
  auto& player_manager = bot.game->player_manager;

  switch (weapon.data.type) {
    case WeaponType::Bomb:
    case WeaponType::ProximityBomb:
    case WeaponType::Thor: {
      // One explosion, seen or not; then a coarse guess at how much it took from each ship it caught.
      u32 now = GetCurrentTick();
      if (now == last_blast_tick && weapon.player_id == last_blast_shooter &&
          weapon.position.DistanceSq(last_blast_position) < 1.0f) {
        return;
      }
      last_blast_tick = now;
      last_blast_shooter = weapon.player_id;
      last_blast_position = weapon.position;

      if (!Notices(weapon.position)) return;

      float damage = settings.BombDamageLevel / 1000.0f;
      float radius = (float)(settings.BombExplodePixels + settings.BombExplodePixels * weapon.data.level) / 16.0f;
      if (radius <= 0.0f) return;

      for (size_t i = 0; i < player_manager.player_count; ++i) {
        Player& player = player_manager.players[i];
        if (player.ship >= 8 || player.id == player_manager.player_id) continue;

        float distance = player.position.Distance(weapon.position) / radius;
        if (distance >= 1.0f) continue;

        // Direct, near or glancing: the middle of each third of the blast's falloff.
        int bucket = (int)(distance * 3.0f);
        if (bucket > 2) bucket = 2;
        Spend(player, damage * (1.0f - (bucket + 0.5f) / 3.0f), UniformSd(damage / 3.0f));
      }
    } break;
    case WeaponType::Bullet:
    case WeaponType::BouncingBullet:
    case WeaponType::Burst: {
      Player* target = event.target;
      if (!target || target->ship >= 8 || target->id == player_manager.player_id) return;
      if (!Notices(target->position)) return;

      float damage;
      if (weapon.data.type == WeaponType::Burst) {
        damage = settings.BurstDamageLevel / 1000.0f;
      } else {
        damage = (settings.BulletDamageLevel + settings.BulletDamageUpgrade * weapon.data.level) / 1000.0f;
        if (weapon.data.shrap > 0) damage *= settings.ShrapnelDamagePercent / 1000.0f;
      }

      // Without exact damage, a hit does a random amount up to its damage, two thirds of it on average.
      if (settings.ExactDamage) {
        Spend(*target, damage, 0.0f);
      } else {
        Spend(*target, damage * 2.0f / 3.0f, damage * 0.24f);
      }
    } break;
    default: {
    } break;
  }
}

void EnergyEstimate::HandleEvent(const PlayerDeathEvent& event) {
  // Respawns full, so start again when next asked.
  beliefs[event.player.id].valid = false;
}

void EnergyEstimate::HandleEvent(const PlayerEnterEvent& event) {
  Spawn(event.player);
}

void EnergyEstimate::HandleEvent(const PlayerFreqAndShipChangeEvent& event) {
  Spawn(event.player);
}

}  // namespace chaos
}  // namespace zero
