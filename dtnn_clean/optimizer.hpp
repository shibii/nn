#pragma once

#include <memory>

#include "hyperparameters.hpp"
#include "optimizable_weights.hpp"

namespace nn {
class Optimizer {
 public:
  Optimizer() = default;
  virtual ~Optimizer() = default;
  virtual void optimize(Hyperparameters hyperparameters) = 0;
  virtual void attach(std::shared_ptr<OptimizableWeights> param) = 0;
};
}  // namespace nn