#pragma once

#include <arrayfire.h>

#include "feed.hpp"

namespace dtnn {
  class LossFunction {
  public:
    LossFunction() = default;
    virtual ~LossFunction() = default;
    virtual af::array error(Feed &f, af::array target) = 0;
    virtual af::array loss(Feed &f, af::array target) = 0;
    virtual af::array output(Feed &f) = 0;
  };
}