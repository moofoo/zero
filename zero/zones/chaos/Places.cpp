#include "Places.h"

#include <zero/Config.h>
#include <zero/game/Logger.h>

#include <algorithm>
#include <sstream>

namespace zero {
namespace chaos {

static std::string Trim(const std::string& text) {
  size_t start = text.find_first_not_of(" \t");
  size_t end = text.find_last_not_of(" \t");
  return start == std::string::npos ? std::string() : text.substr(start, end - start + 1);
}

std::vector<Place> ReadPlaces(Config& config) {
  constexpr const char* kPrefix = "Place.";
  std::vector<Place> places;
  for (const auto& [key, value] : config.GetOrCreateGroup("Chaos").map) {
    if (key.rfind(kPrefix, 0) != 0) continue;
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) parts.push_back(Trim(part));
    Place place;
    place.name = key.substr(std::char_traits<char>::length(kPrefix));
    try {
      if (parts.size() < 4 || parts.size() > 5) throw std::invalid_argument("fields");
      place.center = Vector2f(std::stof(parts[0]), std::stof(parts[1]));
      place.radius = std::stof(parts[2]);
      if (parts.size() == 5 && !parts[4].empty()) place.weight = std::stof(parts[4]);
    } catch (const std::exception&) {
      Log(LogLevel::Warning, "Place.%s: expected x, y, radius, tags[, weight]: %s", place.name.c_str(), value.c_str());
      continue;
    }
    std::stringstream tags(parts[3]);
    std::string tag;
    while (tags >> tag) place.tags.push_back(tag);
    places.push_back(place);
  }
  std::sort(places.begin(), places.end(), [](const Place& a, const Place& b) { return a.name < b.name; });
  return places;
}

float PlaceWant(const Place& place, const TagWeights& weights) {
  float best = 0.0f;
  for (const std::string& tag : place.tags) {
    auto it = weights.find(tag);
    if (it != weights.end()) best = std::max(best, it->second);
  }
  return place.weight * best;
}

std::string WantedTag(const Place& place, const TagWeights& weights) {
  std::string best;
  float best_weight = 0.0f;
  for (const std::string& tag : place.tags) {
    auto it = weights.find(tag);
    if (it != weights.end() && it->second > best_weight) {
      best = tag;
      best_weight = it->second;
    }
  }
  return best;
}

}  // namespace chaos
}  // namespace zero
