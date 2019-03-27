#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../loss_function.hpp"
#include "../feed.hpp"

namespace dtnn {
  class SquaredLoss : public LossFunction {
  public:
    SquaredLoss() = default;
    ~SquaredLoss() = default;
    af::array error(Feed &f, af::array target) const override;
    af::array loss(Feed &f, af::array target) const override;
    af::array output(Feed &f) const override;
    template <class Archive> void serialize(Archive &ar);
  };
}
CEREAL_REGISTER_TYPE(dtnn::SquaredLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::LossFunction, dtnn::SquaredLoss)