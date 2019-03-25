#include "adam.hpp"
#include "../arrayfire_util.hpp"

namespace dtnn {
  Adam::Adam(float learningrate, float decay1, float decay2)
    : learningrate_(learningrate), decay1_(decay1), decay2_(decay2)
  {
  }
  void Adam::optimize() {
    decay1T_ *= decay1_;
    decay2T_ *= decay2_;

    for (auto &state : states_) {
      state.m = decay1_ * state.m + (1.f - decay1_) * state.param->gradient;
      state.v = decay2_ * state.v + (1.f - decay2_) * state.param->gradient.pow(2);

      wb mhat = state.m / (1.f - decay1T_);
      wb vhat = state.v / (1.f - decay2T_);
      wb rootvhat = {
        af::sqrt(util::replace_zeroes(vhat.w)),
        af::sqrt(util::replace_zeroes(vhat.b))
      };

      state.param->weights -= learningrate_ * mhat / (rootvhat);
      state.param->gradient.zero();
    }
  }
  void Adam::attach(std::shared_ptr<OptimizableWeights> param) {
    Adam::OptimizerState state;
    state.param = param;
    state.m.w = af::constant(0.f, param->weights.w.dims());
    state.m.b = af::constant(0.f, param->weights.b.dims());
    state.v.w = af::constant(0.f, param->weights.w.dims());
    state.v.b = af::constant(0.f, param->weights.b.dims());
    states_.push_back(state);
  }
  template <class Archive> void Adam::serialize(Archive &ar) {
    ar(learningrate_, decay1_, decay2_, decay1T_, decay2T_, states_);
  }
}