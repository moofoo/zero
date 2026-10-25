#pragma once

#include <zero/Seed.h>

namespace zero {
namespace chaos {

// The seed for each of the Chaos bot's random generators (zero/Seed.h).
inline u32 ChaosSeed() { return NewSeed(); }

}  // namespace chaos
}  // namespace zero
