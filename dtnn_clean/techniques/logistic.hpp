#pragma once

#include <arrayfire.h>

#include "../propagation_stage.hpp"
#include "../wb.hpp"

namespace dtnn {
  class Logistic : public PropagationStage {
  public:
    ~Logistic() = default;
    Logistic() = default;
    void forward(Pack &p) override;
    void backward(Pack &p) override;

  private:
    af::array activation_;
  };
}