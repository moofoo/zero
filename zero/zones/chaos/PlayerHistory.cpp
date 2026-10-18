#include "PlayerHistory.h"

#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

PlayerHistory::Sample PlayerHistory::Get(const Player& player, u32 delay) const {
  Sample now = {GetCurrentTick(), player.position, player.velocity};

  auto it = tracks.find(player.id);
  if (delay == 0 || it == tracks.end() || it->second.count == 0) return now;

  const Track& track = it->second;
  u32 wanted = now.tick - delay;

  // Newest to oldest: the first sample at or before the wanted tick.
  const Sample* oldest = nullptr;
  for (size_t i = 0; i < track.count; ++i) {
    const Sample& sample = track.samples[(track.next + kSamples - 1 - i) % kSamples];
    if (!TICK_GT(sample.tick, wanted)) return sample;
    oldest = &sample;
  }

  return oldest ? *oldest : now;
}

void PlayerHistory::HandleEvent(const BotController::UpdateEvent& event) {
  u32 tick = GetCurrentTick();
  if (tick == last_tick) return;
  last_tick = tick;

  auto& player_manager = bot.game->player_manager;
  for (size_t i = 0; i < player_manager.player_count; ++i) {
    Player& player = player_manager.players[i];
    if (player.id == player_manager.player_id) continue;

    Track& track = tracks[player.id];
    if (player.ship >= 8) {
      // Not flying: forget where they were, so a respawn isn't aimed at their last life.
      track.count = 0;
      continue;
    }

    track.samples[track.next] = {tick, player.position, player.velocity};
    track.next = (track.next + 1) % kSamples;
    if (track.count < kSamples) ++track.count;
  }
}

}  // namespace chaos
}  // namespace zero
