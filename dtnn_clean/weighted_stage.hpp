#pragma once

#include "propagation_stage.hpp"
#include "optimizable_weights.hpp"

namespace dtnn {
  class WeightedStage : public PropagationStage {
    virtual std::shared_ptr<OptimizableWeights> init(Feed sample) = 0;
  };
}