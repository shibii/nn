#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct Pack {
    af::array signal;
    af::array target;
    af::array loss;
    bool calculateLoss;
  };

  class PropagationStage {
  public:
    PropagationStage() = default;
    virtual ~PropagationStage() = default;
    virtual void forward(Pack &p) = 0;
    virtual void backward(Pack &p) = 0;
  };
}