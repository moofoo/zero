#pragma once

#include <zero/Types.h>

namespace zero {

struct Player;
struct Ship;

// ZERO_INPUTLOG=<folder> (bot-ai ticket 41): a client logs every update's input and its own ship's state afterwards to
// <folder>/<pid>.inp, so the duel simulator can replay a real game and compare. Records:
//   1 (update): tick u32, actions u32, x, y, vx, vy, orientation, energy (f32), ship u8, enter_delay f32, then the
//      ship's upgrades: kShipFields u32s (max energy, recharge, rotation, guns, bombs, thrust, speed, shrapnel, repels,
//      bursts, decoys, thors, bricks, rockets, portals, multifire, capability)
//   2 (one-shot action, such as a toggle): tick u32, action u8
constexpr int kShipFields = 17;
void InputLogUpdate(u32 tick, u32 actions, const Player& self, const Ship& ship);
void ShipFields(const Ship& ship, u32* out);
void SetShipFields(Ship& ship, const u32* in);
void InputLogAction(u32 tick, u8 action);

}  // namespace zero
