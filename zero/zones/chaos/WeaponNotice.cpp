#include "WeaponNotice.h"

#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

#include <algorithm>
#include <vector>

namespace zero {
namespace chaos {

// How often each unnoticed weapon on screen gets a chance to be noticed (ticks), and the chance at full Awareness for
// a bomb at the bot's ship on an otherwise empty screen.
constexpr u32 kRollTicks = 10;
constexpr float kRollChance = 0.5f;
// Bullets (and shrapnel and bursts) are smaller and faster than bombs, mines and thors: harder to see.
constexpr float kBulletVisibility = 0.6f;
// Each other enemy weapon on screen cuts the chance by this much, down to kMinClutter of it.
constexpr float kClutterCut = 0.02f;
constexpr float kMinClutter = 0.5f;

WeaponNotice::WeaponNotice(ZeroBot& bot, float awareness, Vector2f view_extent, u32 reaction_ticks)
    : reaction_ticks(reaction_ticks),
      bot(bot),
      awareness(awareness),
      view_extent(view_extent),
      random(std::random_device{}()) {}

// Fixed for each weapon while it exists: who fired it, when it ends, and what it is. A multifire volley's bullets
// share a key, and are noticed together, as a player sees the spread.
u64 WeaponNotice::Key(const Weapon& weapon) {
  return ((u64)weapon.player_id << 48) ^ ((u64)weapon.end_tick << 8) ^ (u64)weapon.data.type;
}

bool WeaponNotice::Reacting(const Weapon& weapon) const {
  auto at = NoticedAt(weapon);
  return at && !TICK_GT(*at + reaction_ticks, GetCurrentTick());
}

std::optional<u32> WeaponNotice::NoticedAt(const Weapon& weapon) const {
  auto it = noticed.find(Key(weapon));
  if (it == noticed.end()) return {};
  return it->second.tick;
}

std::optional<u32> WeaponNotice::LastBombSeen(u16 player_id) const {
  auto it = last_bomb.find(player_id);
  if (it == last_bomb.end()) return {};
  return it->second;
}

std::optional<u32> WeaponNotice::LastBulletSeen(u16 player_id) const {
  auto it = last_bullet.find(player_id);
  if (it == last_bullet.end()) return {};
  return it->second;
}

float WeaponNotice::Chance(const Weapon& weapon, Vector2f self_position, size_t on_screen) const {
  Vector2f delta = weapon.position - self_position;
  float edge = fmaxf(fabsf(delta.x) / view_extent.x, fabsf(delta.y) / view_extent.y);
  float place = 1.0f - 0.5f * std::min(edge, 1.0f);

  bool big = weapon.data.type == WeaponType::Bomb || weapon.data.type == WeaponType::ProximityBomb ||
             weapon.data.type == WeaponType::Thor;
  float visibility = big ? 1.0f : kBulletVisibility;

  float clutter = std::max(kMinClutter, 1.0f - kClutterCut * (float)(on_screen > 0 ? on_screen - 1 : 0));

  return kRollChance * awareness * visibility * place * clutter;
}

void WeaponNotice::HandleEvent(const BotController::UpdateEvent& event) {
  u32 now = GetCurrentTick();
  if (TICK_GT(next_roll_tick, now)) return;
  next_roll_tick = now + kRollTicks;

  // Forget weapons that are gone.
  for (auto it = noticed.begin(); it != noticed.end();) {
    it = TICK_GT(now, it->second.end_tick) ? noticed.erase(it) : std::next(it);
  }

  Player* self = bot.game->player_manager.GetSelf();
  if (!self || self->ship >= 8) return;

  WeaponManager& weapons = bot.game->weapon_manager;
  std::vector<Weapon*> on_screen;
  for (size_t i = 0; i < weapons.weapon_count; ++i) {
    Weapon& weapon = weapons.weapons[i];
    if (weapon.frequency == self->frequency) continue;
    Vector2f delta = weapon.position - self->position;
    if (fabsf(delta.x) > view_extent.x || fabsf(delta.y) > view_extent.y) continue;
    on_screen.push_back(&weapon);
  }

  std::uniform_real_distribution<float> roll(0.0f, 1.0f);
  for (Weapon* weapon : on_screen) {
    u64 key = Key(*weapon);
    if (noticed.count(key)) continue;
    if (roll(random) >= Chance(*weapon, self->position, on_screen.size())) continue;

    noticed[key] = {now, weapon->end_tick};
    bool bomb = weapon->data.type == WeaponType::Bomb || weapon->data.type == WeaponType::ProximityBomb;
    bool bullet = weapon->data.type == WeaponType::Bullet || weapon->data.type == WeaponType::BouncingBullet;
    if ((bomb && !weapon->data.alternate) || bullet) {
      // When it was fired: its age, from its lifetime and when it ends.
      s32 alive = weapons.GetWeaponTotalAliveTime(weapon->data.type, weapon->data.alternate);
      u32 fired = now - (u32)std::max(0, alive - (s32)TICK_DIFF(weapon->end_tick, now));
      auto& last = bomb ? last_bomb : last_bullet;
      auto it = last.find(weapon->player_id);
      if (it == last.end() || TICK_GT(fired, it->second)) last[weapon->player_id] = fired;
    }
  }
}

}  // namespace chaos
}  // namespace zero
