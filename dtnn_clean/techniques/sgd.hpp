#pragma once

#include "../optimizer.hpp"

namespace dtnn {
  class SGD : public Optimizer {
  public:
    ~SGD() = default;
    SGD(float learningrate);
    void optimize() override;
    void attach(std::shared_ptr<wb> weights, std::shared_ptr<wb> gradient) override;

    float learningrate_;
  private:
    struct State;
    std::vector<State> states_;
  };

  struct SGD::State {
    std::shared_ptr<wb> weights_;
    std::shared_ptr<wb> gradient_;
  };
}