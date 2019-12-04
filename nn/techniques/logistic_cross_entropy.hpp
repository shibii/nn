#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../feed.hpp"
#include "../loss_function.hpp"

namespace nn {
class LogisticCrossEntropy : public LossFunction {
 public:
  LogisticCrossEntropy() = default;
  ~LogisticCrossEntropy() = default;
  af::array error(Feed &f, af::array target) const override;
  af::array loss(Feed &f, af::array target) const override;
  af::array output(Feed &f) const override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);
  af::array logistic(const af::array &input) const;
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::LogisticCrossEntropy);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::LossFunction, nn::LogisticCrossEntropy)