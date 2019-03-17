#pragma once

#include <arrayfire.h>
#include "feed.hpp"

namespace dtnn {
  class LossFunction {
  public:
    LossFunction() = default;
    virtual ~LossFunction() = default;
    virtual void error(Feed &f, af::array target) = 0;
    virtual float loss(af::array target) = 0;
    virtual void output(Feed &f) = 0;
  };
}