#include "momentum.hpp"

namespace nn {
Momentum::Momentum(float decay) : decay_(decay) {}
void Momentum::optimize(Hyperparameters hp) {
  for (auto &state : states_) {
    auto decay_term = state.param->get_decay_deltas(hp.weight_decay);
    state.velocities =
        (1.f - decay_) * state.velocities +
        hp.learningrate * state.param->gradient / (float)hp.batch_size;
    state.param->weights -= state.velocities;
    state.param->apply_weight_decay(decay_term);
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
template <class Archive>
void Momentum::serialize(Archive &ar) {
  ar(decay_, states_);
}
}  // namespace nn