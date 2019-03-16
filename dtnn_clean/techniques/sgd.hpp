#pragma once

#include "../optimizer.hpp"

namespace dtnn {
  class SGD : public Optimizer {
  public:
    SGD() = default;
    ~SGD() = default;
    SGD(float learningrate);
    void optimize() override;
    void attach(std::shared_ptr<OptimizableWeights> param) override;

    float learningrate_;
    std::vector<std::shared_ptr<OptimizableWeights>> params_;
  };
}