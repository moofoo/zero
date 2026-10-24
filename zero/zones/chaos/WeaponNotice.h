#pragma once

#include <zero/BotController.h>
#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/behavior/nodes/WeaponView.h>
#include <zero/game/GameEvent.h>

#include <optional>
#include <random>
#include <unordered_map>

namespace zero {

struct Weapon;
struct ZeroBot;

namespace chaos {

// Which enemy weapons the bot has noticed, as a player would: a weapon on its screen gets a chance to be noticed every
// kRollTicks, so one missed at first can still be noticed as it comes closer. Each chance is kRollChance times
// Awareness, times how visible the weapon is (bullets less than bombs), its place on screen (half as likely at the
// edge as at the bot's ship) and the clutter (less likely with many enemy weapons on screen). The bot reacts to a
// weapon only `reaction_ticks` after noticing it.
struct WeaponNotice : behavior::WeaponNoticer, EventHandler<BotController::UpdateEvent> {
  WeaponNotice(ZeroBot& bot, float awareness, Vector2f view_extent, u32 reaction_ticks);

  // Whether the bot noticed an enemy weapon at least the reaction time ago.
  bool Reacting(const Weapon& weapon) const override;
  // When the bot noticed a weapon, if it has.
  std::optional<u32> NoticedAt(const Weapon& weapon) const;
  // When the bot last saw a player fire a bomb (from the bombs it noticed, and how old each was then), if ever.
  std::optional<u32> LastBombSeen(u16 player_id) const;
  // ... and fire bullets (ticket 33, reading when its guns are ready again).
  std::optional<u32> LastBulletSeen(u16 player_id) const;

  void HandleEvent(const BotController::UpdateEvent& event) override;

  u32 reaction_ticks;

 private:
  static u64 Key(const Weapon& weapon);
  float Chance(const Weapon& weapon, Vector2f self_position, size_t on_screen) const;

  ZeroBot& bot;
  float awareness;
  Vector2f view_extent;  // half the screen, in tiles
  // Each noticed weapon's notice tick and end tick (to forget it once it's gone).
  struct Noticed {
    u32 tick;
    u32 end_tick;
  };
  std::unordered_map<u64, Noticed> noticed;
  std::unordered_map<u16, u32> last_bomb;
  std::unordered_map<u16, u32> last_bullet;
  u32 next_roll_tick = 0;
  std::mt19937 random;
};

}  // namespace chaos
}  // namespace zero
