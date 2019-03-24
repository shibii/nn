#include "momentum.hpp"

namespace dtnn {
  Momentum::Momentum(float learningrate, float decay)
    : learningrate_(learningrate), decay_(decay)
  {
  }
  void Momentum::optimize() {
    for (auto &state : states_) {
      state.velocities = (1.f - decay_) * state.velocities
        + learningrate_ * state.param->gradient;
      state.param->weights -= state.velocities;
      state.param->gradient.zero();
    }
  }
  void Momentum::attach(std::shared_ptr<OptimizableWeights> param) {
    Momentum::OptimizerState state;
    state.param = param;
    state.velocities.w = af::constant(0.f, param->weights.w.dims());
    state.velocities.b = af::constant(0.f, param->weights.b.dims());
    states_.push_back(state);
  }
  template <class Archive> void Momentum::serialize(Archive &ar) {
    ar(learningrate_, decay_, states_);
  }
}