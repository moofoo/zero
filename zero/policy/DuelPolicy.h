#pragma once

#include <zero/Types.h>

#include <string>
#include <vector>

namespace zero {

// A duel policy trained by sim/train.py (bot-ai ticket 42), run by hand: two tanh layers, a GRU and three heads (turn,
// thrust, fire), as train.Policy. Weights come from a file written by sim/export_policy.py. The value head isn't
// needed to play, so it isn't loaded.
struct DuelPolicy {
  static constexpr int kTurns = 5, kThrusts = 3, kFires = 3;
  static constexpr int kLogits = kTurns + kThrusts + kFires;

  // Loads the weights; false (with a log line) if the file is missing or doesn't match the network.
  bool Load(const char* path);

  int obs_size = 0;
  int hidden = 0;

  // One step: the observation (obs_size floats) and the recurrent state (hidden floats, updated in place) give the
  // heads' logits (kLogits: turn, then thrust, then fire).
  void Step(const float* obs, float* state, float* logits) const;

  // The most likely choice of each head from Step's logits.
  static void Greedy(const float* logits, int choice[3]);

 private:
  struct Linear {
    int in = 0, out = 0;
    std::vector<float> weight, bias;  // weight is out x in, row-major
    void Apply(const float* x, float* y) const;
  };
  Linear body0, body1;
  Linear gru_ih, gru_hh;  // 3 x hidden rows each: reset, update, new (PyTorch's order)
  Linear heads[3];
};

// One loaded policy per file, shared by every bot in the process.
const DuelPolicy* LoadSharedPolicy(const std::string& path);

}  // namespace zero
