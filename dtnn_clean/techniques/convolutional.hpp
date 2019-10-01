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
  class Convolutional : public WeightedStage {
  public:
    ~Convolutional() = default;
    Convolutional() = default;
    Convolutional(dim_t size0, dim_t size1, dim_t stride0, dim_t stride1, dim_t pad0, dim_t pad1, dim_t features);
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    std::shared_ptr<OptimizableWeights> init(af::dim4 input) override;
    template <class Archive> void serialize(Archive &ar);

  private:
    std::shared_ptr<OptimizableWeights> param_;
    af::array inputflat_;
    af::dim4 inputdim_;
    af::dim4 reorderdim_;
    af::dim4 convolutiondim_;
    dim_t size0_;
    dim_t size1_;
    dim_t stride0_;
    dim_t stride1_;
    dim_t pad0_;
    dim_t pad1_;
    dim_t features_;
  };
}
CEREAL_REGISTER_TYPE(nn::Convolutional);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::Convolutional)