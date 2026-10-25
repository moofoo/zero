#pragma once

#include <memory>

namespace zero {

struct ZoneController;

namespace chaos {

// A new Chaos zone controller: it handles one bot's zone events from then on (in its event scope, Event.h). zero's own
// bot uses the one made at start-up; the duel simulator makes one for each bot it runs (bot-ai ticket 45).
std::unique_ptr<ZoneController> CreateChaosController();

}  // namespace chaos
}  // namespace zero
