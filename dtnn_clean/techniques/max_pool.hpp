#pragma once

#include <memory>
#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>

#include "../cereal_archives.hpp"
#include "../weightless_stage.hpp"
#include "../feed.hpp"
#include "../optimizable_weights.hpp"

namespace dtnn {
  class MaxPool : public WeightlessStage {
  public:
    ~MaxPool() = default;
    MaxPool() = default;
    MaxPool(dim_t size0, dim_t size1, dim_t stride0, dim_t stride1, dim_t pad0, dim_t pad1);
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    template <class Archive> void serialize(Archive &ar);

  private:
    af::dim4 inputdim_;
    af::dim4 unwrapdim_;
    af::array maxidx_;
    dim_t size0_;
    dim_t size1_;
    dim_t stride0_;
    dim_t stride1_;
    dim_t pad0_;
    dim_t pad1_;
  };
}
CEREAL_REGISTER_TYPE(dtnn::MaxPool);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::PropagationStage, dtnn::MaxPool)