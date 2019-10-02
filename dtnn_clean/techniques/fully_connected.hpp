#pragma once

#include <memory>
#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>

#include "../cereal_archives.hpp"
#include "../weighted_stage.hpp"
#include "../feed.hpp"
#include "../optimizable_weights.hpp"

namespace nn {
  class FullyConnected : public WeightedStage {
  public:
    ~FullyConnected() = default;
    FullyConnected(dim_t units);
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    std::shared_ptr<OptimizableWeights> init(af::dim4 input) override;
    template <class Archive> void serialize(Archive &ar);

  private:
    friend class cereal::access;
    FullyConnected() = default;
    std::shared_ptr<OptimizableWeights> param_;
    dim_t units_;
    af::dim4 inputdim_;
    af::array inputflat_;
  };
}
CEREAL_REGISTER_TYPE(nn::FullyConnected);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::FullyConnected)