#pragma once

#include "feed.hpp"

namespace nn {
  class PropagationStage {
  public:
    PropagationStage() = default;
    virtual ~PropagationStage() = default;
    virtual void forward(Feed &f) = 0;
    virtual void backward(Feed &f) = 0;
  };
}