#pragma once

#include <zero/Math.h>
#include <zero/ZeroBot.h>
#include <zero/behavior/BehaviorTree.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>
#include <zero/game/WeaponManager.h>

namespace zero {
namespace behavior {

// Which weapons a node may react to. Without the blackboard keys below, every weapon counts. A behavior that should
// react only to what a player in its seat would see sets:
// - "view_extent": half its screen's size, in tiles. Only weapons on screen count.
// - "weapon_reaction_ticks": an enemy weapon counts only once it has existed this long.
// - "weapon_notice": the chance of noticing each enemy weapon at all, decided once per weapon.
// Weapons fired by the bot's own team always count.
struct WeaponView {
  WeaponView(ExecuteContext& ctx, const Player& self)
      : settings(ctx.bot->game->connection.settings), center(self.position), self_id(self.id), freq(self.frequency) {
    auto opt_extent = ctx.blackboard.Value<Vector2f>("view_extent");
    if (opt_extent) {
      limited = true;
      extent = *opt_extent;
    }
    reaction_ticks = ctx.blackboard.ValueOr<int>("weapon_reaction_ticks", 0);
    notice = ctx.blackboard.ValueOr<float>("weapon_notice", 1.0f);
  }

  bool Contains(const Weapon& weapon) const {
    if (limited) {
      Vector2f delta = weapon.position - center;
      if (fabsf(delta.x) > extent.x || fabsf(delta.y) > extent.y) return false;
    }

    if (weapon.frequency == freq) return true;
    if (reaction_ticks > 0 && GetAge(weapon) < reaction_ticks) return false;
    if (notice < 1.0f && NoticeRoll(weapon) >= notice) return false;

    return true;
  }

  // How long ago a weapon was fired, in ticks, from its lifetime and when it ends.
  s32 GetAge(const Weapon& weapon) const {
    s32 alive_time = settings.BulletAliveTime;
    if (weapon.data.type == WeaponType::Bomb || weapon.data.type == WeaponType::ProximityBomb ||
        weapon.data.type == WeaponType::Thor) {
      alive_time = weapon.data.alternate ? settings.MineAliveTime : settings.BombAliveTime;
    }
    return alive_time - TICK_DIFF(weapon.end_tick, GetCurrentTick());
  }

  // A number in [0, 1) fixed for each weapon and bot, so a weapon noticed (or missed) stays that way.
  float NoticeRoll(const Weapon& weapon) const {
    u32 h = (weapon.end_tick * 0x9E3779B1u) ^ ((u32)weapon.player_id << 16) ^ self_id;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return (h >> 8) / (float)(1 << 24);
  }

  const ArenaSettings& settings;
  Vector2f center;
  u16 self_id;
  u16 freq;
  Vector2f extent;
  bool limited = false;
  int reaction_ticks = 0;
  float notice = 1.0f;
};

}  // namespace behavior
}  // namespace zero
