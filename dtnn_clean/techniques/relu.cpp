#include "relu.hpp"

namespace dtnn {
  void ReLU::forward(Feed &f) {
    input_ = f.signal;
    f.signal = (input_ > 0.f) * input_;
  }
  void ReLU::backward(Feed &f) {
    f.signal = (input_ > 0.f) * f.signal;
  }
  template <class Archive> void ReLU::serialize(Archive &ar) {
  }
}