#pragma once

#include <arrayfire.h>

#include "../feed.hpp"
#include "../propagation_stage.hpp"
#include "../wb.hpp"

namespace dtnn {
  class FullyConnected : public PropagationStage {
  public:
    ~FullyConnected() = default;
    FullyConnected() = default;
    FullyConnected(std::shared_ptr<wb> weights, std::shared_ptr<wb> gradient);
    FullyConnected(std::shared_ptr<wb> weights);
    void forward(Feed &f) override;
    void backward(Feed &f) override;

    std::shared_ptr<wb> weights_;
    std::shared_ptr<wb> gradient_;
  private:
    af::dim4 inputdim_;
    af::array inputflat_;
  };
}