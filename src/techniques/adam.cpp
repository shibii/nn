#include "adam.hpp"
#include "../arrayfire_util.hpp"

namespace nn {
Adam::Adam(float decay1, float decay2)
    : decay1_(decay1), decay2_(decay2), decay1T_(1), decay2T_(1) {}
void Adam::optimize(Hyperparameters hp) {
  decay1T_ *= decay1_;
  decay2T_ *= decay2_;

  for (auto &state : states_) {
    auto decay_term = state.param->get_decay_deltas(hp.weight_decay);
    auto avg_gradient = state.param->gradient / (float)hp.batch_size;
    state.m = decay1_ * state.m + (1.f - decay1_) * avg_gradient;
    state.v = decay2_ * state.v + (1.f - decay2_) * avg_gradient.pow(2);
    wb mhat = state.m / (1.f - decay1T_);
    wb vhat = state.v / (1.f - decay2T_);
    // sqrt is numerically stabilized internally
    state.param->weights -= hp.learningrate * mhat / vhat.sqrt();
    state.param->apply_weight_decay(decay_term);
    state.param->gradient.reset();
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
template <class Archive>
void Adam::serialize(Archive &ar) {
  ar(decay1_, decay2_, decay1T_, decay2T_, states_);
}
}  // namespace nn