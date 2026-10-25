#include "InputLog.h"

#include <stdio.h>
#include <stdlib.h>
#include <zero/game/Player.h>
#include <zero/game/ShipController.h>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace zero {

static FILE* InputLogFile() {
  static FILE* file = nullptr;
  static bool checked = false;
  if (!checked) {
    checked = true;
    if (const char* folder = getenv("ZERO_INPUTLOG")) {
      char path[512];
      snprintf(path, sizeof(path), "%s/%d.inp", folder, (int)getpid());
      file = fopen(path, "wb");
    }
  }
  return file;
}

void ShipFields(const Ship& s, u32* out) {
  const u32 v[kShipFields] = {s.energy,  s.recharge, s.rotation, s.guns,  s.bombs,   s.thrust,      s.speed,
                              s.shrapnel, s.repels,  s.bursts,   s.decoys, s.thors,  s.bricks,      s.rockets,
                              s.portals,  (u32)s.multifire, (u32)s.capability};
  for (int i = 0; i < kShipFields; ++i) out[i] = v[i];
}

void SetShipFields(Ship& s, const u32* in) {
  s.energy = in[0], s.recharge = in[1], s.rotation = in[2], s.guns = in[3], s.bombs = in[4], s.thrust = in[5];
  s.speed = in[6], s.shrapnel = in[7], s.repels = in[8], s.bursts = in[9], s.decoys = in[10], s.thors = in[11];
  s.bricks = in[12], s.rockets = in[13], s.portals = in[14], s.multifire = in[15] != 0;
  s.capability = (ShipCapabilityFlags)in[16];
}

void InputLogUpdate(u32 tick, u32 actions, const Player& self, const Ship& ship) {
  FILE* f = InputLogFile();
  if (!f) return;
  u8 type = 1;
  float values[6] = {self.position.x, self.position.y, self.velocity.x, self.velocity.y, self.orientation, self.energy};
  u8 ship_type = (u8)self.ship;
  float enter_delay = self.enter_delay;
  fwrite(&type, 1, 1, f);
  fwrite(&tick, 4, 1, f);
  fwrite(&actions, 4, 1, f);
  fwrite(values, 4, 6, f);
  fwrite(&ship_type, 1, 1, f);
  fwrite(&enter_delay, 4, 1, f);
  u32 fields[kShipFields];
  ShipFields(ship, fields);
  fwrite(fields, 4, kShipFields, f);
}

void InputLogAction(u32 tick, u8 action) {
  FILE* f = InputLogFile();
  if (!f) return;
  u8 type = 2;
  fwrite(&type, 1, 1, f);
  fwrite(&tick, 4, 1, f);
  fwrite(&action, 1, 1, f);
}

}  // namespace zero
