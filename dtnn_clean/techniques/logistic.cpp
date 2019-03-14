#include "logistic.hpp"

namespace dtnn {
  void Logistic::forward(Pack &p) {
    activation_ = 1.f / (1.f + af::exp(-p.signal));
    p.signal = activation_;
  }
  void Logistic::backward(Pack &p) {

  }
}