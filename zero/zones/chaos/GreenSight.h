#pragma once

#include <zero/BotController.h>
#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/game/Game.h>
#include <zero/game/GameEvent.h>

#include <random>
#include <unordered_map>
#include <vector>

namespace zero {

struct ZeroBot;

namespace chaos {

// Which greens the bot has noticed, as a player would (ticket 32): a green on its screen or radar gets a chance to be
// noticed every kRollTicks, Awareness times kScreenChance on screen or kRadarChance only on radar, so a novice notices
// greens later than a veteran and can pass one before noticing it. Without a skill level, every green on screen or
// radar is noticed at once. Also logs the greens the bot picks up, by what it went for.
struct GreenSight : EventHandler<BotController::UpdateEvent>, EventHandler<GreenPickupEvent> {
  static constexpr u32 kRollTicks = 25;
  static constexpr float kScreenChance = 0.6f;
  static constexpr float kRadarChance = 0.2f;

  GreenSight(ZeroBot& bot, float awareness, bool instant, Vector2f view_extent);

  // The greens the bot has noticed that are still there.
  const std::vector<PrizeGreen>& Noticed() const { return noticed_greens; }

  // What the bot is flying to a green for ("detour" or "need"), to log its pickup by; cleared on arrival.
  void Aim(Vector2f position, const char* why) {
    aim = position;
    aim_why = why;
  }

  void HandleEvent(const BotController::UpdateEvent& event) override;
  void HandleEvent(const GreenPickupEvent& event) override;

 private:
  static u32 Key(Vector2f position) { return ((u32)position.x << 16) | (u32)position.y; }

  ZeroBot& bot;
  float awareness;
  bool instant;
  Vector2f view_extent;  // half the screen, in tiles
  std::unordered_map<u32, u32> noticed;  // by tile, the green's end tick
  std::vector<PrizeGreen> noticed_greens;
  Vector2f aim;
  const char* aim_why = nullptr;
  u32 last_tick = 0;
  u32 next_roll_tick = 0;
  std::mt19937 random;
};

}  // namespace chaos
}  // namespace zero
