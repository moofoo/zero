#pragma once

#include <zero/Math.h>
#include <zero/Types.h>
#include <zero/game/GameEvent.h>

#include <random>

namespace zero {

struct Player;
struct ZeroBot;

namespace chaos {

// What a Chaos bot believes about other players' energy, built only from what a player in its seat could notice.
// The arena doesn't send other players' energy, so this is a guess that carries its uncertainty, and the uncertainty
// is what makes committing to an attack a wager.
//
// Each player's estimate is a mean and a standard deviation:
// - On spawn: the ship's average of initial and maximum energy. Chaos spawns ships with random prizes, so its real
//   maximum could be anywhere in that range, and the deviation covers it.
// - Over time, while on the bot's screen: recharge at the average of initial and maximum recharge, up to that
//   average maximum, with the deviation growing by the spread of recharge rates.
// - Over time, while off screen: they could be fighting anyone, so the estimate relaxes towards a typical energy
//   (60% of the average maximum) with the widest deviation, over a few seconds.
// - Shots and hits: only those on the bot's screen, and each noticed with probability `attention`. A noticed shot
//   costs its weapon's fire cost (with a little noise, as a player judges it); a noticed hit costs a coarse amount
//   (a bullet; a bomb direct, near or glancing) and widens the deviation. Knowing they miss some, the bot counts
//   each noticed event for 1/attention of itself, with the extra deviation that guess carries, so low attention
//   makes the estimate noisier rather than optimistic.
// Energy the server does send (a teammate in a SeeEnergy = Team arena) is used exactly while it's fresh.
struct EnergyEstimate : EventHandler<WeaponFireEvent>,
                        EventHandler<WeaponHitEvent>,
                        EventHandler<PlayerDeathEvent>,
                        EventHandler<PlayerEnterEvent>,
                        EventHandler<PlayerFreqAndShipChangeEvent> {
  struct Reading {
    float energy;
    float sd;
  };

  EnergyEstimate(ZeroBot& bot, float attention, Vector2f view_extent);

  // The current estimate of a player's energy.
  Reading Get(Player& player);

  void HandleEvent(const WeaponFireEvent& event) override;
  void HandleEvent(const WeaponHitEvent& event) override;
  void HandleEvent(const PlayerDeathEvent& event) override;
  void HandleEvent(const PlayerEnterEvent& event) override;
  void HandleEvent(const PlayerFreqAndShipChangeEvent& event) override;

 private:
  struct Belief {
    float mean = 0.0f;
    float sd = 0.0f;
    u32 tick = 0;  // when it was last brought up to date
    bool valid = false;
  };

  ZeroBot& bot;
  float attention;
  Vector2f view_extent;  // half the screen, in tiles
  std::mt19937 random;
  Belief beliefs[65535];
  // The last bomb explosion handled: a bomb sends a hit event for each ship it touches, but explodes once.
  u32 last_blast_tick = 0;
  u16 last_blast_shooter = 0xFFFF;
  Vector2f last_blast_position;

  // Resets a player's belief to a freshly spawned ship.
  void Spawn(Player& player);
  // Brings a player's belief up to the current tick: recharge and growing uncertainty.
  Belief& Advance(Player& player);
  // Whether a position is on the bot's screen.
  bool InView(Vector2f position);
  // Whether the bot sees and notices something happening at this position.
  bool Notices(Vector2f position);
  // Takes a noticed amount of energy off a player's belief.
  void Spend(Player& player, float amount, float amount_sd);
};

}  // namespace chaos
}  // namespace zero
