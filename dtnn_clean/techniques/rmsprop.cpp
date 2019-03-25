#include "rmsprop.hpp"
#include "../arrayfire_util.hpp"

namespace dtnn {
  RMSprop::RMSprop(float learningrate, float decay)
    : learningrate_(learningrate), decay_(decay)
  {
  }
  void RMSprop::optimize() {
    for (auto &state : states_) {
      state.rms = (1.f - decay_) * state.rms
        + decay_ * state.param->gradient.pow(2);

      af::array rootw = af::sqrt(util::replace_zeroes(state.rms.w));
      af::array rootb = af::sqrt(util::replace_zeroes(state.rms.b));
      wb root = { rootw, rootb };

      state.param->weights -= learningrate_ * (state.param->gradient / root);
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