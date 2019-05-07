#include "adagrad.hpp"
#include "../arrayfire_util.hpp"

namespace dtnn {
  Adagrad::Adagrad(float learningrate)
    : learningrate_(learningrate)
  {
  }
  void Adagrad::optimize(unsigned int batch_size) {
    for (auto &state : states_) {
      auto avg_gradient = state.param->gradient / batch_size;
      state.sum_of_squared_grad += avg_gradient.pow(2);
      state.param->weights -= learningrate_ * avg_gradient / state.sum_of_squared_grad.sqrt();
      state.param->gradient.zero();
    }
  }
  void Adagrad::attach(std::shared_ptr<OptimizableWeights> param) {
    Adagrad::OptimizerState state;
    state.param = param;
    state.sum_of_squared_grad.w = af::constant(0.f, param->weights.w.dims());
    state.sum_of_squared_grad.b = af::constant(0.f, param->weights.b.dims());
    states_.push_back(state);
  }
  template <class Archive> void Adagrad::serialize(Archive &ar) {
    ar(learningrate_, states_);
  }
}