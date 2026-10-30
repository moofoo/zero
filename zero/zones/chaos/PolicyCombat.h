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

struct EnergyEstimate;

// The learned duel policy as a combat node (bot-ai ticket 42): from when the bot's target comes on its screen until it
// is off both screen and radar, the policy flies the ship (turn, thrust, fire) as it did in the simulator, and the
// rest of the combat tree waits. It sees what the simulator showed it (FairObserve: taken every kObsEvery ticks, acted
// on kObsDelayTicks late), decides ten times a second, and taps or holds the turn as it chose. Fails, leaving the tree
// to fly, without a policy, a target, or with the target out of view; its memory starts over on a new target or after
// a pause. A policy trained with the opponent's energy estimate (ticket 46) gets it after the dial, from the bot's
// own EnergyEstimate, as of the observation it acts on.
struct PolicyCombatNode : public behavior::BehaviorNode {
  static constexpr u32 kStepTicks = 10;  // a decision every 10 ticks, as the simulator's steps
  static constexpr u32 kTapTicks = 3;    // a tapped turn is held this long (duelsim.DuelEnv.TAP_TICKS)
  static constexpr u32 kPauseTicks = 50;  // off for longer than this, the policy starts over

  PolicyCombatNode(const DuelPolicy* policy, float aggression, EnergyEstimate* estimate)
      : policy(policy), aggression(aggression), estimate(estimate) {}

  behavior::ExecuteResult Execute(behavior::ExecuteContext& ctx) override;

 private:
  const DuelPolicy* policy;
  float aggression;  // the dial input after the observation, for a policy trained with one (ticket 50)
  EnergyEstimate* estimate;
  u16 target_id = 0xFFFF;
  u32 last_tick = 0;
  bool active = false;
  // (tick, observation and the target's energy estimate), the latest last
  std::deque<std::pair<u32, std::vector<float>>> history;
  std::vector<float> state;
  u32 decision_tick = 0;
  int choice[3] = {2, 1, 0};  // turn none, thrust none, fire none

  void Restart(u32 tick);
};

}  // namespace chaos
}  // namespace zero
