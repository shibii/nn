#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../feed.hpp"
#include "../weightless_stage.hpp"

namespace nn {
class Logistic : public WeightlessStage {
 public:
  ~Logistic() = default;
  Logistic() = default;
  void forward(Feed &f) override;
  void backward(Feed &f) override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);

  af::array activation_;
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::Logistic);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::Logistic)