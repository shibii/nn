#include "sgd.hpp"
#include <iostream>

namespace dtnn {
  SGD::SGD(float learningrate) : learningrate_(learningrate) {
  }
  void SGD::optimize() {
    for (auto param : params_) {
      param->weights -= learningrate_ * param->gradient;
      param->gradient.zero();
    }
  }
  void SGD::attach(std::shared_ptr<OptimizableWeights> param) {
    params_.push_back(param);
  }
}