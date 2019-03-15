#include "sgd.hpp"

namespace dtnn {
  SGD::SGD(float learningrate) : learningrate_(learningrate) {

  }
  void SGD::optimize() {
    for (auto state : states_) {
      *state.weights_ -= learningrate_ * *state.gradient_;
    }
  }
  void SGD::attach(std::shared_ptr<wb> weights, std::shared_ptr<wb> gradient) {
    states_.push_back({ weights, gradient });
  }
}