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
class FullyConnected : public WeightedStage {
 public:
  ~FullyConnected() = default;
  FullyConnected(dim_t units);
  void forward(Feed &f) override;
  void backward(Feed &f) override;
  std::shared_ptr<OptimizableWeights> init(const af::dim4 input) override;

 private:
  FullyConnected() = default;
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);

  std::shared_ptr<OptimizableWeights> param_;
  dim_t units_;
  af::dim4 inputdim_;
  af::array inputflat_;
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::FullyConnected);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::FullyConnected)