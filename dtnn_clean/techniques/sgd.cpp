#include "sgd.hpp"

namespace dtnn {
  SGD::SGD(float learningrate) : learningrate_(learningrate) {
  }
  void SGD::optimize(unsigned int batch_size) {
    for (auto &param : params_) {
      param->weights -= learningrate_ * param->gradient / batch_size;
      param->gradient.zero();
    }
  }
  void SGD::attach(std::shared_ptr<OptimizableWeights> param) {
    params_.push_back(param);
  }
  template <class Archive> void SGD::serialize(Archive &ar) {
    ar(learningrate_, params_);
  }
}