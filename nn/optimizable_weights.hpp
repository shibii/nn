#pragma once

#include "wb.hpp"

namespace nn {
struct OptimizableWeights {
  wb weights;
  wb gradient;

  af::array get_decay_deltas(const float decay) const {
    return weights.w * decay;
  }
  void apply_weight_decay(const af::array deltas) {
    weights.w -= deltas;
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(weights, gradient);
  }
};
}  // namespace nn