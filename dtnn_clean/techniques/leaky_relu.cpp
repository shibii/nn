#include "leaky_relu.hpp"

namespace nn {
  LeakyReLU::LeakyReLU(float leak)
    : leak_(leak)
  {
  }
  void LeakyReLU::forward(Feed &f) {
    input_ = f.signal;
    f.signal = (input_ > 0.f) * input_ + (input_ <= 0.f) * input_ * leak_;
  }
  void LeakyReLU::backward(Feed &f) {
    f.signal = (input_ > 0.f) * f.signal + (input_ <= 0.f) * f.signal * leak_;
  }
  template <class Archive> void LeakyReLU::serialize(Archive &ar) {
    ar(leak_);
  }
}