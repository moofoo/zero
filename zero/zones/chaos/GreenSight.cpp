#include "GreenSight.h"

#include <zero/CombatLogger.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

GreenSight::GreenSight(ZeroBot& bot, float awareness, bool instant, Vector2f view_extent)
    : bot(bot), awareness(awareness), instant(instant), view_extent(view_extent), random(std::random_device{}()) {}

void GreenSight::HandleEvent(const BotController::UpdateEvent& event) {
  u32 now = GetCurrentTick();
  if (now == last_tick) return;
  last_tick = now;

  Game& game = *bot.game;
  Player* self = game.player_manager.GetSelf();
  noticed_greens.clear();
  if (!self || self->ship >= 8) {
    noticed.clear();
    return;
  }

  bool roll = instant || !TICK_GT(next_roll_tick, now);
  if (roll) next_roll_tick = now + kRollTicks;
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);

  std::unordered_map<u32, u32> still;
  for (size_t i = 0; i < game.green_count; ++i) {
    const PrizeGreen& green = game.greens[i];
    if (green.end_tick == 0) continue;  // picked up, removed next update
    u32 key = Key(green.position);
    auto it = noticed.find(key);
    if (it == noticed.end()) {
      Vector2f delta = green.position - self->position;
      bool on_screen = fabsf(delta.x) <= view_extent.x && fabsf(delta.y) <= view_extent.y;
      if (!on_screen && !game.radar.InRadarView(green.position)) continue;
      if (!roll) continue;
      if (!instant && chance(random) >= awareness * (on_screen ? kScreenChance : kRadarChance)) continue;
    }
    still[key] = green.end_tick;
    noticed_greens.push_back(green);
  }
  noticed.swap(still);
}

void GreenSight::HandleEvent(const GreenPickupEvent& event) {
  const char* why = "incidental";
  if (aim_why && event.green.position.DistanceSq(aim) < 1.5f * 1.5f) why = aim_why;
  if (bot.combat_logger) bot.combat_logger->WriteGreen("pickup", event.green.position, why, 0.0f, event.green.prize_id);
  aim_why = nullptr;
}

}  // namespace chaos
}  // namespace zero
