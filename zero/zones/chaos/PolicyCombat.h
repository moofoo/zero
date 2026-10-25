#pragma once

#include <zero/Types.h>
#include <zero/behavior/BehaviorTree.h>
#include <zero/policy/FairObservation.h>

#include <deque>
#include <utility>
#include <vector>

namespace zero {

struct DuelPolicy;

namespace chaos {

// The learned duel policy as a combat node (bot-ai ticket 42): from when the bot's target comes on its screen until it
// is off both screen and radar, the policy flies the ship (turn, thrust, fire) as it did in the simulator, and the
// rest of the combat tree waits. It sees what the simulator showed it (FairObserve: taken every kObsEvery ticks, acted
// on kObsDelayTicks late), decides ten times a second, and taps or holds the turn as it chose. Fails, leaving the tree
// to fly, without a policy, a target, or with the target out of view; its memory starts over on a new target or after
// a pause.
struct PolicyCombatNode : public behavior::BehaviorNode {
  static constexpr u32 kStepTicks = 10;  // a decision every 10 ticks, as the simulator's steps
  static constexpr u32 kTapTicks = 3;    // a tapped turn is held this long (duelsim.DuelEnv.TAP_TICKS)
  static constexpr u32 kPauseTicks = 50;  // off for longer than this, the policy starts over

  explicit PolicyCombatNode(const DuelPolicy* policy) : policy(policy) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override;

 private:
  const DuelPolicy* policy;
  u16 target_id = 0xFFFF;
  u32 last_tick = 0;
  bool active = false;
  std::deque<std::pair<u32, std::vector<float>>> history;  // (tick, observation), the latest last
  std::vector<float> state;
  u32 decision_tick = 0;
  int choice[3] = {2, 1, 0};  // turn none, thrust none, fire none

  void Restart(u32 tick);
};

}  // namespace chaos
}  // namespace zero
