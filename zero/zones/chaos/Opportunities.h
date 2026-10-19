#pragma once

#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/game/GameEvent.h>

#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zero {

struct Player;
struct ZeroBot;

namespace chaos {

// Mistakes an opponent makes that a skilled player punishes, as the bot sees them (ticket 24). Each opportunity is
// taken with the bot's spotting chance (its Judgement), decided once per opportunity, and logged as a tactic:
// - A fire window: an enemy the bot sees fire a bomb can't fire anything again for its ship's bomb delay. The bot
//   notices the bomb as it notices any weapon (its Awareness), and only on its screen.
// - A slow target: an enemy flying slower than kSlowSpeed is easier to hit, by factors measured from Chaos games.
//   Spotted once per slowdown, on screen.
// It also keeps what the target lock needs for battle blasting: which other enemies it has noticed trading hits on
// its screen, and when the bot itself last fought.
struct Opportunities : EventHandler<WeaponFireEvent>, EventHandler<WeaponSelfDamageEvent>, EventHandler<WeaponHitEvent> {
  // Below this speed (tiles/s) a target counts as slow.
  static constexpr float kSlowSpeed = 3.0f;

  Opportunities(ZeroBot& bot, float spotting, float notice, Vector2f view_extent);

  // Seconds left of a fire window the bot is punishing against this player (0 without one).
  float WindowSeconds(const Player& player);
  // How much likelier a shot of this kind at this distance (tiles) is to hit the player, if the bot has spotted it
  // flying slowly (1 otherwise).
  float SlowFactor(const Player& player, bool bomb, float distance);

  // Takes an opportunity with the bot's spotting chance.
  bool Spot() { return Roll(spotting); }
  // When an enemy last hit the bot, if one has.
  std::optional<u32> LastHitTick() const { return last_hit_tick; }
  // Seconds since the bot last hit or was hit by an enemy (a large number if never).
  float SecondsSinceFight() const;
  // Two enemies fighting each other: both among the candidates, close together, having traded hits the bot noticed on
  // its screen just now, the nearer at battle-blasting range from the bot. The pair comes lower id first.
  std::optional<std::pair<u16, u16>> FindBattle(const std::vector<Player*>& candidates, const Player& self);
  // Whether two enemies are still trading hits: either has hit the other recently.
  bool BattleGoing(u16 a, u16 b) const;

  void HandleEvent(const WeaponFireEvent& event) override;
  void HandleEvent(const WeaponSelfDamageEvent& event) override;
  void HandleEvent(const WeaponHitEvent& event) override;

 private:
  struct Slowdown {
    bool slow = false;     // flying slowly when last looked at
    bool spotted = false;  // and the bot took the opportunity
    u32 seen_tick = 0;     // when it was last looked at
  };

  ZeroBot& bot;
  float spotting;
  float notice;
  Vector2f view_extent;  // half the screen, in tiles
  std::mt19937 random;
  std::unordered_map<u16, u32> window_end;
  std::unordered_map<u16, Slowdown> slowdowns;
  std::optional<u32> last_hit_tick;
  std::optional<u32> last_dealt_tick;
  // When each enemy last hit each other enemy (noticed, on the bot's screen), keyed by PairKey(shooter, target).
  std::unordered_map<u32, u32> enemy_hits;

  // When one enemy last hit another, if the bot noticed it.
  std::optional<u32> LastHit(u16 shooter, u16 target) const;
  // Whether a tick is within this many ticks of now.
  static bool Recent(std::optional<u32> tick, u32 ticks);

  bool OnScreen(const Player& player) const;
  bool Roll(float chance);
  // Whether the bot is punishing this player's slowdown now: rolls once as a slowdown starts.
  bool SpottedSlow(const Player& player);
};

// A key for an ordered pair of player ids.
inline u32 PairKey(u16 first, u16 second) {
  return ((u32)first << 16) | second;
}

}  // namespace chaos
}  // namespace zero
