#include "rmsprop.hpp"
#include "../arrayfire_util.hpp"

namespace dtnn {
  RMSprop::RMSprop(float learningrate, float decay)
    : learningrate_(learningrate), decay_(decay)
  {
  }
  void RMSprop::optimize(unsigned int batch_size) {
    for (auto &state : states_) {
      auto avg_gradient = state.param->gradient / batch_size;

      state.rms = (1.f - decay_) * state.rms
        + decay_ * avg_gradient.pow(2);

      state.param->weights -= learningrate_ * (avg_gradient / state.rms.sqrt());
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
    ar(learningrate_, decay_, states_);
  }
}