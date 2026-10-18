#pragma once

#include <zero/BotController.h>
#include <zero/Math.h>
#include <zero/Types.h>

#include <array>
#include <unordered_map>

namespace zero {

struct Player;
struct ZeroBot;

namespace chaos {

// Where other players have been over the last moment, so a bot with a reaction time can act on where it saw them
// rather than where they are this instant.
struct PlayerHistory : EventHandler<BotController::UpdateEvent> {
  struct Sample {
    u32 tick = 0;
    Vector2f position;
    Vector2f velocity;
  };

  // A little over half a second of ticks: longer than any reaction time a skill level sets.
  static constexpr size_t kSamples = 64;

  explicit PlayerHistory(ZeroBot& bot) : bot(bot) {}

  // Where a player was `delay` ticks ago: the latest sample at least that old, else the oldest one kept, else where
  // they are now.
  Sample Get(const Player& player, u32 delay) const;

  void HandleEvent(const BotController::UpdateEvent& event) override;

 private:
  struct Track {
    std::array<Sample, kSamples> samples;
    size_t next = 0;
    size_t count = 0;
  };

  ZeroBot& bot;
  std::unordered_map<u16, Track> tracks;
  u32 last_tick = 0;
};

}  // namespace chaos
}  // namespace zero
