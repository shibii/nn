#pragma once

#include <memory>

#include "optimizable_weights.hpp"
#include "hyperparameters.hpp"

namespace nn {
  class Optimizer {
  public:
    Optimizer() = default;
    virtual ~Optimizer() = default;
    virtual void optimize(Hyperparameters hyperparameters) = 0;
    virtual void attach(std::shared_ptr<OptimizableWeights> param) = 0;
  };
}