#pragma once

#include <zero/BotController.h>
#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/game/GameEvent.h>

#include <optional>
#include <unordered_map>

namespace zero {

struct Player;
struct ZeroBot;

namespace chaos {

// Which enemies the bot can see, as a player in its seat would: on its screen at once, unless cloaked; on its radar
// (Continuum's radar window around the bot) once it has noticed the dot, after `radar_notice_ticks`, unless stealthed
// or cloaked. XRadar sees through stealth and cloak. An enemy out of sight is remembered where it was last seen.
struct EnemySight : EventHandler<BotController::UpdateEvent>, EventHandler<PlayerLeaveEvent> {
  struct Sighting {
    u32 tick;
    Vector2f position;
    Vector2f velocity;
  };

  EnemySight(ZeroBot& bot, Vector2f view_extent, u32 radar_notice_ticks);

  // Whether the bot sees an enemy this tick.
  bool Sees(const Player& player) const;
  // Where and when the bot last saw a player, if it has.
  std::optional<Sighting> LastSeen(u16 player_id) const;

  void HandleEvent(const BotController::UpdateEvent& event) override;
  void HandleEvent(const PlayerLeaveEvent& event) override;

 private:
  struct Track {
    bool seen = false;
    std::optional<u32> on_radar_since;
    std::optional<Sighting> last_seen;
  };

  bool OnScreen(const Player& self, const Player& player) const;

  ZeroBot& bot;
  Vector2f view_extent;  // half the screen, in tiles
  u32 radar_notice_ticks;
  std::unordered_map<u16, Track> tracks;
  u32 last_tick = 0;
};

}  // namespace chaos
}  // namespace zero
