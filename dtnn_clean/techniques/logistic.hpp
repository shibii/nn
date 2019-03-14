#pragma once

#include <arrayfire.h>

#include "../feed.hpp"
#include "../propagation_stage.hpp"
#include "../wb.hpp"

namespace dtnn {
  class Logistic : public PropagationStage {
  public:
    ~Logistic() = default;
    Logistic() = default;
    void forward(Feed &f) override;
    void backward(Feed &f) override;

  private:
    af::array activation_;
  };
}