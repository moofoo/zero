#pragma once

#include <stdio.h>
#include <zero/Types.h>
#include <zero/game/GameEvent.h>

namespace zero {

struct ZeroBot;

// Writes this bot's combat events, one JSON object per line, for measuring how bots fight (chaos-offline's
// `chaos.py stats`). Every event carries the server tick, so logs from separate bots line up. It only records
// the bot's own point of view: its shots, the damage it takes, its kills and deaths, and how it handled enemy
// bombs near it. Enabled with --events <path>.
struct CombatLogger : EventHandler<WeaponFireEvent>,
                      EventHandler<WeaponSelfDamageEvent>,
                      EventHandler<WeaponHitEvent>,
                      EventHandler<PlayerDeathEvent>,
                      EventHandler<PlayerFreqAndShipChangeEvent> {
  CombatLogger(ZeroBot& bot, const char* path);
  ~CombatLogger();

  bool IsOpen() const { return file != nullptr; }

  // Call once per game tick. Writes the bot's position, velocity and energy twice a second.
  void Update();

  // The bot's estimate of its target's energy (mean and standard deviation), and zero's heuristic for comparison.
  // Call as often as the estimate is read; it's written at most twice a second.
  void WriteEstimate(const char* target, float energy, float sd, float heuristic);

  // The bot's target changed (null: no target), and why.
  void WriteTarget(const char* target, const char* reason);

  // The commit decision's mode changed, and the win chance it judged.
  void WriteMode(const char* mode, float win_chance);

  void HandleEvent(const WeaponFireEvent& event) override;
  void HandleEvent(const WeaponSelfDamageEvent& event) override;
  void HandleEvent(const WeaponHitEvent& event) override;
  void HandleEvent(const PlayerDeathEvent& event) override;
  void HandleEvent(const PlayerFreqAndShipChangeEvent& event) override;

 private:
  ZeroBot& bot;
  FILE* file = nullptr;
  u32 next_state_tick = 0;
  u32 next_estimate_tick = 0;
  // The last bomb that damaged the bot (by shooter and tick), so its explosion isn't also counted as a near miss.
  // Not a Weapon pointer: the weapon array is compacted, so another bomb can later sit at the same address.
  u16 last_damaging_shooter = 0xFFFF;
  u32 last_damage_tick = 0;

  u32 ServerTick();
  void WriteSpawn(int ship, int freq);
};

}  // namespace zero
