#pragma once

#include <zero/Types.h>

#include <optional>
#include <random>

namespace zero {

// The seed for a bot's random generators: from the system, unless g_Seed is set. The duel simulator sets it from an
// environment's seed while it works on a bot there (bot-ai ticket 45), so its runs repeat.
inline std::optional<u32> g_Seed;

inline u32 NewSeed() {
  if (g_Seed) return (*g_Seed)++;
  return std::random_device{}();
}

}  // namespace zero
