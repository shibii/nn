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
    void error(Feed &f, af::array target) override;
    float loss(af::array target) override;
    void output(Feed &f);
    template<class Archive> void serialize(Archive & archive);
    af::array output_;
  };
}
CEREAL_REGISTER_TYPE(dtnn::SquaredLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::LossFunction, dtnn::SquaredLoss);