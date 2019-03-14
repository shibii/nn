#pragma once

#include <arrayfire.h>
#include "feed.hpp"

namespace dtnn {
  class LossFunction {
  public:
    LossFunction() = default;
    virtual ~LossFunction() = default;
    virtual void error(Feed &f) = 0;
  };
}