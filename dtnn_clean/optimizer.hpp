#pragma once

#include <arrayfire.h>
#include "wb.hpp"

namespace dtnn {
  class OptimizerState {
  public:
    std::shared_ptr<wb> weights_;
    std::shared_ptr<wb> gradient_;
    std::shared_ptr<wb> state_;
  };

  class Optimizer {
  public:
    Optimizer() = default;
    virtual ~Optimizer() = default;
    virtual void optimize() = 0;
    virtual void attach(std::shared_ptr<wb> weights, std::shared_ptr<wb> gradient) = 0;
  };
}