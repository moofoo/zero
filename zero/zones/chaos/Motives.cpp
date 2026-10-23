#include "Motives.h"

#include <zero/CombatLogger.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>

namespace zero {
namespace chaos {

const char* to_string(Motive motive) {
  switch (motive) {
    case Motive::Grudge:
      return "grudge";
    case Motive::Coward:
      return "coward";
    default:
      return "roam";
  }
}

std::optional<Motive> ParseMotive(std::string_view name) {
  if (name == "roam") return Motive::Roam;
  if (name == "grudge") return Motive::Grudge;
  if (name == "coward") return Motive::Coward;
  return {};
}

float DeathPull(Motive motive) {
  switch (motive) {
    case Motive::Grudge:
      return 1.0f;
    case Motive::Coward:
      return -1.0f;
    default:
      return -0.3f;  // everyone keeps a little away from where they last died
  }
}

std::optional<u16> DeathMemory::Grudge() const {
  if (grudge == 0xFFFF || TICK_GT(GetCurrentTick(), grudge_until)) return {};
  return grudge;
}

void DeathMemory::HandleEvent(const PlayerDeathEvent& event) {
  Player* self = bot.game->player_manager.GetSelf();
  if (!self) return;
  u32 now = GetCurrentTick();
  if (event.player.id == self->id) {
    last = Death{now, event.player.position, event.killer.id};
    returned = false;
    if (motive == Motive::Grudge && event.killer.id != self->id) {
      bool renewed = grudge == event.killer.id && Grudge().has_value();
      grudge = event.killer.id;
      grudge_until = now + kGrudgeTicks;
      if (bot.combat_logger) bot.combat_logger->WriteTactic(renewed ? "grudge renewed" : "grudge", event.killer.name);
    }
  } else if (event.killer.id == self->id && Grudge() && event.player.id == grudge) {
    if (bot.combat_logger) bot.combat_logger->WriteTactic("grudge settled", event.player.name);
    grudge = 0xFFFF;
  }
}

}  // namespace chaos
}  // namespace zero
