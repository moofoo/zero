#include "EnemySight.h"

#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

EnemySight::EnemySight(ZeroBot& bot, Vector2f view_extent, u32 radar_notice_ticks)
    : bot(bot), view_extent(view_extent), radar_notice_ticks(radar_notice_ticks) {}

bool EnemySight::Sees(const Player& player) const {
  auto it = tracks.find(player.id);
  return it != tracks.end() && it->second.seen;
}

std::optional<EnemySight::Sighting> EnemySight::LastSeen(u16 player_id) const {
  auto it = tracks.find(player_id);
  if (it == tracks.end()) return {};
  return it->second.last_seen;
}

bool EnemySight::OnScreen(const Player& self, const Player& player) const {
  Vector2f delta = player.position - self.position;
  return fabsf(delta.x) <= view_extent.x && fabsf(delta.y) <= view_extent.y;
}

void EnemySight::HandleEvent(const BotController::UpdateEvent& event) {
  u32 now = GetCurrentTick();
  if (now == last_tick) return;
  last_tick = now;

  Game& game = *bot.game;
  Player* self = game.player_manager.GetSelf();
  if (!self) return;
  bool xradar = (self->togglables & Status_XRadar) != 0;

  for (size_t i = 0; i < game.player_manager.player_count; ++i) {
    Player& player = game.player_manager.players[i];
    if (player.id == self->id) continue;
    Track& track = tracks[player.id];

    bool in_game = self->ship < 8 && player.ship < 8 && !player.IsRespawning() && player.position != Vector2f(0, 0);
    bool cloaked = !xradar && (player.togglables & Status_Cloak);
    bool stealthed = !xradar && (player.togglables & Status_Stealth);
    bool on_screen = in_game && !cloaked && OnScreen(*self, player);
    // Cloak hides a ship on screen, stealth on radar (as zero's own radar and screen do).
    bool on_radar = in_game && !stealthed && game.radar.InRadarView(player.position);

    if (on_radar || on_screen) {
      if (!track.on_radar_since) track.on_radar_since = now;
    } else {
      track.on_radar_since.reset();
    }
    // A dot on radar is noticed after a while; once seen, an enemy stays seen while it's on screen or radar.
    bool noticed = track.on_radar_since && TICK_DIFF(now, *track.on_radar_since) >= (s32)radar_notice_ticks;
    track.seen = on_screen || (on_radar && (track.seen || noticed));
    if (track.seen) track.last_seen = Sighting{now, player.position, player.velocity};
  }
}

void EnemySight::HandleEvent(const PlayerLeaveEvent& event) { tracks.erase(event.player.id); }

}  // namespace chaos
}  // namespace zero
