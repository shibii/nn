#include "sgd.hpp"

namespace nn {
void SGD::optimize(Hyperparameters hp) {
  for (auto &param : params_) {
    auto decay_term = param->weights.w * hp.weight_decay;
    param->weights -= hp.learningrate * param->gradient / hp.batch_size;
    param->weights.w -= decay_term;
    param->gradient.zero();
  }
}
void SGD::attach(std::shared_ptr<OptimizableWeights> param) {
  params_.push_back(param);
}
template <class Archive>
void SGD::serialize(Archive &ar) {
  ar(params_);
}
}  // namespace nn