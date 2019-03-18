#pragma once

#include <arrayfire.h>

#include "../serialization.hpp"
#include "../loss_function.hpp"
#include "../feed.hpp"

namespace dtnn {
  class SquaredLoss : public LossFunction {
  public:
    ~SquaredLoss() = default;
    SquaredLoss() = default;
    af::array error(Feed &f, af::array target) override;
    af::array loss(Feed &f, af::array target) override;
    af::array output(Feed &f);
    template<class Archive> void serialize(Archive & archive);
  };
}
CEREAL_REGISTER_TYPE(dtnn::SquaredLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::LossFunction, dtnn::SquaredLoss);