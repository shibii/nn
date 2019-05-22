#include "rmsprop.hpp"
#include "../arrayfire_util.hpp"

namespace dtnn {
  RMSprop::RMSprop(float decay)
    : decay_(decay)
  {
  }
  void RMSprop::optimize(Hyperparameters hp) {
    for (auto &state : states_) {
      auto decay_term = state.param->weights.w * hp.weight_decay;
      auto avg_gradient = state.param->gradient / (float)hp.batch_size;

      state.rms = (1.f - decay_) * state.rms
        + decay_ * avg_gradient.pow(2);

      state.param->weights -= hp.learningrate * (avg_gradient / state.rms.sqrt());
      state.param->weights.w -= decay_term;
      state.param->gradient.zero();
    }
  }
  void RMSprop::attach(std::shared_ptr<OptimizableWeights> param) {
    RMSprop::OptimizerState state;
    state.param = param;
    state.rms.w = af::constant(0.f, param->weights.w.dims());
    state.rms.b = af::constant(0.f, param->weights.b.dims());
    states_.push_back(state);
  }
  template <class Archive> void RMSprop::serialize(Archive &ar) {
    ar(decay_, states_);
  }
}