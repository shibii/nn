#pragma once

#include <memory>

#include "propagation_stage.hpp"
#include "optimizable_weights.hpp"
#include "feed.hpp"

namespace dtnn {
  class WeightedStage : public PropagationStage {
  public:
    virtual std::shared_ptr<OptimizableWeights> init(af::dim4 input) = 0;
  };
}