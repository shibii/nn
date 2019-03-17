#include "squared_loss.hpp"

namespace dtnn {
  void SquaredLoss::error(Feed &f, af::array target) {
    output_ = f.signal;
    f.signal = -(target - f.signal);
  }
  float SquaredLoss::loss(af::array target) {
    return af::sum<float>(0.5 * af::pow(target - output_, 2));
  }
  void SquaredLoss::output(Feed &f) {
    output_ = f.signal;
  }
}