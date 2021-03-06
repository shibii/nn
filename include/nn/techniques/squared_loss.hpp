#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../feed.hpp"
#include "../loss_function.hpp"

namespace nn {
class SquaredLoss : public LossFunction {
 public:
  SquaredLoss() = default;
  ~SquaredLoss() = default;
  af::array error(Feed &f, const af::array target) const override;
  af::array loss(Feed &f, const af::array target) const override;
  af::array output(Feed &f) const override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::SquaredLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::LossFunction, nn::SquaredLoss)