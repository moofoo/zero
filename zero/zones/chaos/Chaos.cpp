#include <zero/BotController.h>
#include <zero/ZeroBot.h>
#include <zero/game/Logger.h>
#include <zero/zones/ZoneController.h>
#include <zero/zones/svs/TerrierBehavior.h>
#include <zero/zones/svs/WarbirdBehavior.h>

namespace zero {
namespace chaos {

// Chaos is a kill-them-all zone on SVS-style settings, so it reuses the SVS dogfighting behaviors.
struct ChaosController : ZoneController {
  bool IsZone(Zone zone) override { return zone == Zone::Chaos; }

  void CreateBehaviors(const char* arena_name) override;
};

static ChaosController controller;

void ChaosController::CreateBehaviors(const char* arena_name) {
  Log(LogLevel::Info, "Registering Chaos behaviors.");

  bot->bot_controller->energy_tracker.estimate_type = EnergyHeuristicType::Average;

  auto& repo = bot->bot_controller->behaviors;

  repo.Add("warbird", std::make_unique<svs::WarbirdBehavior>());
  repo.Add("terrier", std::make_unique<svs::TerrierBehavior>());

  SetBehavior("warbird");
}

}  // namespace chaos
}  // namespace zero
