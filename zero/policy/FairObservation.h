#pragma once

#include <zero/Types.h>

namespace zero {

struct Game;
struct WeaponData;

// The fair observation (ticket 42): what a player in the seat could see, kFairSize floats, scaled to about -1..1.
// Positions and velocities are in the ship's frame, as (ahead, right) of its heading, so "the enemy is to the right"
// is one number. Own ship (8): velocity ahead, right (/20 tiles/s), heading x, y (on the map), energy share, ticks
// until bullet (/100) and bomb (/150) are ready, alive. Opponent (8): on screen (the 1920x1080 screen: 60 x 33.75
// tiles either side), only on radar, offset ahead, right (/60), and on screen only, relative velocity ahead, right
// (/20) and its heading ahead, right. Enemy weapons on screen, the nearest kFairWeapons (8 each): offset ahead, right
// (/60), relative velocity ahead, right (/40), bullet, bomb, other, present. Walls: kFairRays rays from the ship, the
// first straight ahead and then clockwise, the distance to the first wall (/kRayLength), 1 for none.
constexpr int kFairWeapons = 16;
constexpr int kFairRays = 16;
constexpr float kRayLength = 30.0f;
constexpr int kFairSize = 8 + 8 + 8 * kFairWeapons + kFairRays;
constexpr float kScreenX = 60.0f, kScreenY = 33.75f;
// Fair observations are taken every kObsEvery ticks and acted on kObsDelayTicks late (the reaction delay).
constexpr u32 kObsEvery = 5;
constexpr u32 kObsDelayTicks = 15;

// The fair observation of the client `g`'s own ship against the player `other_pid`, at `tick`: kFairSize floats.
// Shared by the duel simulator (bot-ai ticket 42) and zero's policy combat node, so both see the same numbers.
void FairObserve(Game& g, u16 other_pid, u32 tick, float* out);

// A weapon's class in fair observations and the simulator's hit records: 0 bullet, 1 bomb, 2 other (shrapnel, mines,
// bursts and the rest).
int WeaponClass(const WeaponData& data);

}  // namespace zero
