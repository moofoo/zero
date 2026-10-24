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
    case Motive::Brawler:
      return "brawler";
    default:
      return "roam";
  }
}

std::optional<Motive> ParseMotive(std::string_view name) {
  if (name == "roam") return Motive::Roam;
  if (name == "grudge") return Motive::Grudge;
  if (name == "coward") return Motive::Coward;
  if (name == "brawler") return Motive::Brawler;
  return {};
}

float DeathPull(Motive motive) {
  switch (motive) {
    case Motive::Grudge:
      return 1.0f;
    case Motive::Coward:
      return -1.0f;
    case Motive::Brawler:
      return 0.0f;
    default:
      return -0.3f;  // everyone keeps a little away from where they last died
  }
}

float EngageRange(Motive motive) {
  switch (motive) {
    case Motive::Grudge:
      return 25.0f;
    case Motive::Coward:
      return 15.0f;
    case Motive::Brawler:
      return 1e6f;
    default:
      return 40.0f;
  }
}

TagWeights MotiveWeights(Motive motive) {
  switch (motive) {
    case Motive::Brawler:
      return {{"center", 3.0f}, {"chokepoint", 1.2f}, {"corridor", 0.8f}, {"base", 0.5f}, {"greens", 0.4f},
              {"quiet", 0.2f}};
    case Motive::Coward:
      return {{"quiet", 1.5f}, {"greens", 1.3f}, {"base", 1.2f}, {"corridor", 0.8f}, {"chokepoint", 0.3f},
              {"center", 0.1f}};
    default:  // roam, grudge
      return {{"quiet", 1.5f}, {"greens", 1.3f}, {"corridor", 1.2f}, {"chokepoint", 1.0f}, {"base", 0.8f},
              {"center", 0.5f}};
  }
}

std::pair<u32, u32> TagLinger(const std::string& tag) {
  if (tag == "center") return {500, 1000};  // waiting for a fight
  if (tag == "quiet" || tag == "greens") return {300, 600};
  if (tag == "corridor") return {0, 50};  // passing through
  return {200, 400};  // chokepoint and base, until mining and safe zones are built
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
