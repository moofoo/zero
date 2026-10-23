#pragma once

#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/game/GameEvent.h>
#include <zero/zones/chaos/Places.h>

#include <optional>
#include <string>

namespace zero {

struct ZeroBot;

namespace chaos {

// A bot's motive (ticket 09): what it wants from the game, which steers whom it fights and where it goes, while the
// commit decision still decides each moment.
// - Roam: wander the map's places, preferring quieter ones (the default).
// - Grudge: after dying, hunt the killer, going back toward where the bot died or the killer was last seen, locking
//   onto it in sight and committing against it more readily, until the bot kills it or kGrudgeTicks pass (every
//   death to the same killer renews it).
// - Coward: after dying, keep away from where it died for a while.
enum class Motive { Roam, Grudge, Coward };

const char* to_string(Motive motive);
std::optional<Motive> ParseMotive(std::string_view name);

// How strongly a motive pulls a bot toward (+1) or pushes it away from (-1) where it last died, while roaming.
float DeathPull(Motive motive);

// What a bot remembers of its last death and the grudge it holds.
struct DeathMemory : EventHandler<PlayerDeathEvent> {
  static constexpr u32 kGrudgeTicks = 18000;  // three minutes

  DeathMemory(ZeroBot& bot, Motive motive) : bot(bot), motive(motive) {}

  void HandleEvent(const PlayerDeathEvent& event) override;

  struct Death {
    u32 tick = 0;
    Vector2f position;
    u16 killer = 0xFFFF;
  };
  // The bot's last death, if it has died.
  const std::optional<Death>& Last() const { return last; }
  // The enemy the bot holds a grudge against now, if any.
  std::optional<u16> Grudge() const;

  // Hunting (grudge): set once the bot has been back to where it died, so it roams from there.
  bool returned = false;

  ZeroBot& bot;
  Motive motive;

 private:
  std::optional<Death> last;
  u16 grudge = 0xFFFF;
  u32 grudge_until = 0;
};

}  // namespace chaos
}  // namespace zero
