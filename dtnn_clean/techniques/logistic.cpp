#include "logistic.hpp"

namespace dtnn {
  void Logistic::forward(Feed &f) {
    activation_ = 1.f / (1.f + af::exp(-f.signal));
    f.signal = activation_;
  }
  void Logistic::backward(Feed &f) {
    f.signal = f.signal * (activation_ * (1.f - activation_));
  }
  template<class Archive> void Logistic::serialize(Archive & archive)
  {
  }
}