#pragma once

#include <arrayfire.h>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>

#include "../cereal_archives.hpp"
#include "../feed.hpp"
#include "../optimizable_weights.hpp"
#include "../weighted_stage.hpp"

namespace nn {
class Convolutional : public WeightedStage {
 public:
  ~Convolutional() = default;
  Convolutional(dim_t size0, dim_t size1, dim_t stride0, dim_t stride1,
                dim_t pad0, dim_t pad1, dim_t features);
  void forward(Feed &f) override;
  void backward(Feed &f) override;
  std::shared_ptr<OptimizableWeights> init(af::dim4 input) override;
  template <class Archive>
  void serialize(Archive &ar);

 private:
  friend class cereal::access;
  Convolutional() = default;
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
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::Convolutional);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::Convolutional)