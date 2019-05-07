#pragma once

#include <memory>

#include "optimizable_weights.hpp"

namespace dtnn {
  class Optimizer {
  public:
    Optimizer() = default;
    virtual ~Optimizer() = default;
    virtual void optimize(unsigned int batch_size) = 0;
    virtual void attach(std::shared_ptr<OptimizableWeights> param) = 0;
  };
}