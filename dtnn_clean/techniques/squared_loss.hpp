#pragma once

#include "../feed.hpp"
#include "../loss_function.hpp"

namespace dtnn {
  class SquaredLoss : public LossFunction {
  public:
    ~SquaredLoss() = default;
    SquaredLoss() = default;
    void error(Feed &f, af::array target) override;
  };
}