#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../feed.hpp"
#include "../weightless_stage.hpp"

namespace nn {
class ReLU : public WeightlessStage {
 public:
  ~ReLU() = default;
  ReLU() = default;
  void forward(Feed &f) override;
  void backward(Feed &f) override;
  template <class Archive>
  void serialize(Archive &ar);

 private:
  af::array input_;
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::ReLU);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::ReLU)