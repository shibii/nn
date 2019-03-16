#pragma once

#include <arrayfire.h>
#include <cereal/access.hpp>

#include "../feed.hpp"
#include "../weighted_stage.hpp"
#include "../wb.hpp"

namespace dtnn {
  class FullyConnected : public WeightedStage {
  public:
    ~FullyConnected() = default;
    FullyConnected() = default;
    FullyConnected(dim_t units);
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    std::shared_ptr<OptimizableWeights> init(Feed sample);

    std::shared_ptr<OptimizableWeights> param_;
    dim_t units_;
  private:
    af::dim4 inputdim_;
    af::array inputflat_;
  };
}