#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../feed.hpp"
#include "../weightless_stage.hpp"

namespace nn {
class LReL : public WeightlessStage {
 public:
  ~LReL() = default;
  LReL(float leak = 0.01f);
  void forward(Feed &f) override;
  void backward(Feed &f) override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);

  af::array input_;
  float leak_;
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::LReL);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::LReL)