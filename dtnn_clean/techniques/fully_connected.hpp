#pragma once

#include <arrayfire.h>

#include "../propagation_stage.hpp"
#include "../wb.hpp"

namespace dtnn {
  class FullyConnected : public PropagationStage {
  public:
    ~FullyConnected() = default;
    FullyConnected() = default;
    FullyConnected(std::shared_ptr<wb> weights);
    void forward(Pack &p) override;
    void backward(Pack &p) override;

    std::shared_ptr<wb> weights_;
    std::shared_ptr<wb> gradient_;
  private:
    af::dim4 inputdim_;
    af::array inputflat_;
  };
}