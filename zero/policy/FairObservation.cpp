#include "FairObservation.h"

#include <zero/game/Game.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace zero {

int WeaponClass(const WeaponData& data) {
  WeaponType type = data.type;
  if (type == WeaponType::Bullet || type == WeaponType::BouncingBullet) return data.shrap > 0 ? 2 : 0;
  if (type == WeaponType::Bomb || type == WeaponType::ProximityBomb) return data.alternate ? 2 : 1;
  if (type == WeaponType::Thor) return 1;
  return 2;
}

void FairObserve(Game& g, u16 other_pid, u32 tick, float* out) {
  Player* self = g.player_manager.GetSelf();
  for (int k = 0; k < kFairSize; ++k) out[k] = 0.0f;
  if (!self) return;
  Ship& ship = g.ship_controller.ship;
  int n = 0;
  Vector2f heading = self->GetHeading();
  Vector2f right(-heading.y, heading.x);  // y grows down the map, so this is clockwise of the heading
  auto ahead_of = [&](Vector2f v) { return v.x * heading.x + v.y * heading.y; };
  auto right_of = [&](Vector2f v) { return v.x * right.x + v.y * right.y; };
  bool alive = self->ship < 8 && self->enter_delay <= 0.0f;
  out[n++] = ahead_of(self->velocity) / 20.0f, out[n++] = right_of(self->velocity) / 20.0f;
  out[n++] = heading.x, out[n++] = heading.y;
  out[n++] = ship.energy > 0 ? self->energy / (float)ship.energy : 0.0f;
  out[n++] = std::max<s32>(0, TICK_DIFF(ship.next_bullet_tick, tick)) / 100.0f;
  out[n++] = std::max<s32>(0, TICK_DIFF(ship.next_bomb_tick, tick)) / 150.0f;
  out[n++] = alive ? 1.0f : 0.0f;
  Player* them = g.player_manager.GetPlayerById(other_pid);
  if (them && them->ship < 8 && them->enter_delay <= 0.0f) {
    Vector2f d = them->position - self->position;
    bool on_screen = fabsf(d.x) <= kScreenX && fabsf(d.y) <= kScreenY;
    bool on_radar = !on_screen && g.radar.InRadarView(them->position);
    if (on_screen || on_radar) {
      out[n] = on_screen ? 1.0f : 0.0f, out[n + 1] = on_radar ? 1.0f : 0.0f;
      out[n + 2] = ahead_of(d) / 60.0f, out[n + 3] = right_of(d) / 60.0f;
      if (on_screen) {
        Vector2f dv = them->velocity - self->velocity;
        Vector2f h = them->GetHeading();
        out[n + 4] = ahead_of(dv) / 20.0f, out[n + 5] = right_of(dv) / 20.0f;
        out[n + 6] = ahead_of(h), out[n + 7] = right_of(h);
      }
    }
  }
  n += 8;
  std::vector<std::pair<float, const Weapon*>> near;
  WeaponManager& wm = g.weapon_manager;
  for (size_t k = 0; k < wm.weapon_count; ++k) {
    const Weapon& w = wm.weapons[k];
    if (w.frequency == self->frequency) continue;
    Vector2f d = w.position - self->position;
    if (fabsf(d.x) > kScreenX || fabsf(d.y) > kScreenY) continue;
    near.push_back({d.LengthSq(), &w});
  }
  std::sort(near.begin(), near.end(), [](auto& x, auto& y) { return x.first < y.first; });
  for (int k = 0; k < kFairWeapons; ++k, n += 8) {
    if (k >= (int)near.size()) continue;
    const Weapon& w = *near[k].second;
    Vector2f d = w.position - self->position, dv = w.velocity - self->velocity;
    int c = WeaponClass(w.data);
    out[n] = ahead_of(d) / 60.0f, out[n + 1] = right_of(d) / 60.0f;
    out[n + 2] = ahead_of(dv) / 40.0f, out[n + 3] = right_of(dv) / 40.0f;
    out[n + 4] = c == 0, out[n + 5] = c == 1, out[n + 6] = c == 2, out[n + 7] = 1.0f;
  }
  auto& map = g.GetMap();
  for (int r = 0; r < kFairRays; ++r) {
    float angle = r * 2.0f * 3.14159265f / kFairRays;
    Vector2f dir = heading * cosf(angle) + right * sinf(angle);
    CastResult cast = map.Cast(self->position, dir, kRayLength, self->frequency);
    out[n++] = cast.hit ? cast.distance / kRayLength : 1.0f;
  }
}

}  // namespace zero
