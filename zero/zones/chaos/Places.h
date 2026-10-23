#pragma once

#include <zero/Math.h>
#include <zero/Types.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace zero {

struct Config;

namespace chaos {

// A place on the map that bots choose where to go from (bot-ai ticket 09): a circle with tags, written in the [Chaos]
// section as Place.<name> = x, y, radius, tag tag, weight, and edited with chaos-offline's `chaos.py map`.
struct Place {
  std::string name;
  Vector2f center;
  float radius = 0.0f;
  std::vector<std::string> tags;
  float weight = 1.0f;
};

// How much a motive wants each tag; a place counts its most wanted tag, and untagged places none.
using TagWeights = std::unordered_map<std::string, float>;

// The places in a config's [Chaos] section, sorted by name so every bot sees them in the same order. Malformed ones
// are skipped with a warning in the log.
std::vector<Place> ReadPlaces(Config& config);

// A place's draw for a motive: its weight times its most wanted tag's weight.
float PlaceWant(const Place& place, const TagWeights& weights);

// A roamer's tag weights (ticket 09's slice 1): quieter places and the ways between them before the centre.
TagWeights RoamerWeights();

}  // namespace chaos
}  // namespace zero
