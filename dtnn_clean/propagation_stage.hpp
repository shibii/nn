#pragma once

#include <arrayfire.h>
#include "feed.hpp"

namespace dtnn {
  class PropagationStage {
  public:
    PropagationStage() = default;
    virtual ~PropagationStage() = default;
    virtual void forward(Feed &f) = 0;
    virtual void backward(Feed &f) = 0;
  };
}