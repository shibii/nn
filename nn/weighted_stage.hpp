#pragma once

#include <memory>

#include "feed.hpp"
#include "optimizable_weights.hpp"
#include "propagation_stage.hpp"

namespace nn {
class WeightedStage : public PropagationStage {
 public:
  virtual std::shared_ptr<OptimizableWeights> init(af::dim4 input) = 0;
};
}  // namespace nn