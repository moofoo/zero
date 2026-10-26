#include "PolicyCombat.h"

#include <zero/BotController.h>
#include <zero/ZeroBot.h>
#include <zero/game/Clock.h>
#include <zero/game/Game.h>
#include <zero/policy/DuelPolicy.h>

#include <cmath>

namespace zero {
namespace chaos {

using namespace behavior;

void PolicyCombatNode::Restart(u32 tick) {
  history.clear();
  state.assign(policy->hidden, 0.0f);
  decision_tick = tick - kStepTicks;  // decide at once
  choice[0] = 2, choice[1] = 1, choice[2] = 0;
}

ExecuteResult PolicyCombatNode::Execute(ExecuteContext& ctx) {
  if (!policy) return ExecuteResult::Failure;
  Game& game = *ctx.bot->game;
  Player* self = game.player_manager.GetSelf();
  auto& input = ctx.bot->bot_controller->input;
  auto opt_target = ctx.blackboard.Value<Player*>("nearest_target");
  Player* target = opt_target ? *opt_target : nullptr;
  if (!self || !input || !target || self->ship >= 8 || self->enter_delay > 0.0f) {
    active = false;
    return ExecuteResult::Failure;
  }
  // It takes over when the target comes on screen, and keeps the ship while the target stays on screen or radar (as in
  // training, where it always flew); handing back at the screen's edge left a fleeing ship to the tree, which turned.
  Vector2f d = target->position - self->position;
  bool on_screen = fabsf(d.x) <= kScreenX && fabsf(d.y) <= kScreenY;
  bool keep = active && target->id == target_id && game.radar.InRadarView(target->position);
  if (!on_screen && !keep) {
    active = false;
    return ExecuteResult::Failure;
  }

  u32 tick = GetCurrentTick();
  if (!active || target->id != target_id || TICK_DIFF(tick, last_tick) > (s32)kPauseTicks) Restart(tick);
  active = true, target_id = target->id, last_tick = tick;

  if (history.empty() || TICK_DIFF(tick, history.back().first) >= (s32)kObsEvery) {
    std::vector<float> obs(kFairSize);
    FairObserve(game, target->id, tick, obs.data());
    history.push_back({tick, std::move(obs)});
    while (history.size() > 16) history.pop_front();
  }

  if (TICK_DIFF(tick, decision_tick) >= (s32)kStepTicks) {
    // The latest observation at least the reaction delay old; none yet is all zeros, as in the simulator.
    std::vector<float> obs(kFairSize, 0.0f);
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
      if (TICK_DIFF(tick, it->first) >= (s32)kObsDelayTicks) {
        obs = it->second;
        break;
      }
    }
    obs.resize(policy->obs_size, aggression);  // a dial policy's extra input (train.py --dial)
    float logits[DuelPolicy::kLogits];
    policy->Step(obs.data(), state.data(), logits);
    DuelPolicy::Greedy(logits, choice);
    decision_tick = tick;
  }

  // Turn 0 left, 1 tap left, 2 none, 3 tap right, 4 right; thrust 0 forward, 1 none, 2 reverse; fire 0 none, 1 bullet,
  // 2 bomb (duelsim.DuelEnv.choice).
  bool tapping = TICK_DIFF(tick, decision_tick) < (s32)kTapTicks;
  if (choice[0] == 0 || (choice[0] == 1 && tapping)) input->SetAction(InputAction::Left, true);
  if (choice[0] == 4 || (choice[0] == 3 && tapping)) input->SetAction(InputAction::Right, true);
  if (choice[1] == 0) input->SetAction(InputAction::Forward, true);
  if (choice[1] == 2) input->SetAction(InputAction::Backward, true);
  if (choice[2] == 1) input->SetAction(InputAction::Bullet, true);
  if (choice[2] == 2) input->SetAction(InputAction::Bomb, true);
  ctx.bot->bot_controller->actuator.skip_once = true;
  return ExecuteResult::Success;
}

}  // namespace chaos
}  // namespace zero
