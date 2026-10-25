#include "DuelPolicy.h"

#include <zero/game/Logger.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

namespace zero {

namespace {

struct Tensor {
  std::vector<u32> shape;
  std::vector<float> data;
};

// The file: "ZDP1", a u32 count, then each tensor as a u32 name length, the name, a u32 dimension count, the u32
// dimensions and the float32 values (sim/export_policy.py).
bool ReadTensors(const char* path, std::map<std::string, Tensor>& out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  u32 count = 0;
  bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "ZDP1", 4) == 0 && fread(&count, 4, 1, f) == 1;
  for (u32 i = 0; ok && i < count; ++i) {
    u32 name_size = 0, dims = 0;
    ok = fread(&name_size, 4, 1, f) == 1 && name_size < 256;
    std::string name(name_size, '\0');
    ok = ok && fread(name.data(), 1, name_size, f) == name_size && fread(&dims, 4, 1, f) == 1 && dims <= 4;
    Tensor t;
    t.shape.resize(dims);
    size_t size = 1;
    for (u32 d = 0; ok && d < dims; ++d) {
      ok = fread(&t.shape[d], 4, 1, f) == 1;
      size *= t.shape[d];
    }
    if (!ok) break;
    t.data.resize(size);
    ok = fread(t.data.data(), 4, size, f) == size;
    out[name] = std::move(t);
  }
  fclose(f);
  return ok;
}

float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace

void DuelPolicy::Linear::Apply(const float* x, float* y) const {
  for (int o = 0; o < out; ++o) {
    const float* w = weight.data() + (size_t)o * in;
    float sum = bias[o];
    for (int i = 0; i < in; ++i) sum += w[i] * x[i];
    y[o] = sum;
  }
}

bool DuelPolicy::Load(const char* path) {
  std::map<std::string, Tensor> t;
  if (!ReadTensors(path, t)) {
    Log(LogLevel::Error, "Policy: couldn't read %s", path);
    return false;
  }
  auto linear = [&](const char* weight, const char* bias, Linear& l) {
    auto w = t.find(weight), b = t.find(bias);
    if (w == t.end() || b == t.end() || w->second.shape.size() != 2) return false;
    l.out = (int)w->second.shape[0], l.in = (int)w->second.shape[1];
    if (b->second.data.size() != (size_t)l.out) return false;
    l.weight = w->second.data, l.bias = b->second.data;
    return true;
  };
  bool ok = linear("body.0.weight", "body.0.bias", body0) && linear("body.2.weight", "body.2.bias", body1) &&
            linear("gru.weight_ih_l0", "gru.bias_ih_l0", gru_ih) && linear("gru.weight_hh_l0", "gru.bias_hh_l0", gru_hh) &&
            linear("heads.0.weight", "heads.0.bias", heads[0]) && linear("heads.1.weight", "heads.1.bias", heads[1]) &&
            linear("heads.2.weight", "heads.2.bias", heads[2]);
  if (ok) {
    obs_size = body0.in, hidden = body1.out;
    ok = body1.in == body0.out && gru_ih.in == hidden && gru_ih.out == 3 * hidden && gru_hh.in == hidden &&
         gru_hh.out == 3 * hidden && heads[0].out == kTurns && heads[1].out == kThrusts && heads[2].out == kFires;
    for (const Linear& h : heads) ok = ok && h.in == hidden;
  }
  if (!ok) Log(LogLevel::Error, "Policy: %s doesn't hold the duel policy network", path);
  return ok;
}

void DuelPolicy::Step(const float* obs, float* state, float* logits) const {
  std::vector<float> a(hidden), x(hidden), gi(3 * hidden), gh(3 * hidden);
  body0.Apply(obs, a.data());
  for (float& v : a) v = tanhf(v);
  body1.Apply(a.data(), x.data());
  for (float& v : x) v = tanhf(v);
  // PyTorch's GRU: r and z gate the old state; n is the candidate, h' = (1 - z) n + z h.
  gru_ih.Apply(x.data(), gi.data());
  gru_hh.Apply(state, gh.data());
  for (int k = 0; k < hidden; ++k) {
    float r = Sigmoid(gi[k] + gh[k]);
    float z = Sigmoid(gi[hidden + k] + gh[hidden + k]);
    float n = tanhf(gi[2 * hidden + k] + r * gh[2 * hidden + k]);
    state[k] = (1.0f - z) * n + z * state[k];
  }
  heads[0].Apply(state, logits);
  heads[1].Apply(state, logits + kTurns);
  heads[2].Apply(state, logits + kTurns + kThrusts);
}

void DuelPolicy::Greedy(const float* logits, int choice[3]) {
  const int sizes[3] = {kTurns, kThrusts, kFires};
  for (int h = 0, start = 0; h < 3; start += sizes[h], ++h) {
    choice[h] = 0;
    for (int k = 1; k < sizes[h]; ++k) {
      if (logits[start + k] > logits[start + choice[h]]) choice[h] = k;
    }
  }
}

const DuelPolicy* LoadSharedPolicy(const std::string& path) {
  static std::mutex mutex;
  static std::map<std::string, std::unique_ptr<DuelPolicy>> policies;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = policies.find(path);
  if (it == policies.end()) {
    auto policy = std::make_unique<DuelPolicy>();
    if (!policy->Load(path.c_str())) policy.reset();
    it = policies.emplace(path, std::move(policy)).first;
  }
  return it->second.get();
}

}  // namespace zero
