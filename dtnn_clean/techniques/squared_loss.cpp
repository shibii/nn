#include "squared_loss.hpp"

namespace dtnn {
  void SquaredLoss::error(Feed &f, af::array target) {
    f.signal = -(target - f.signal);
  }
}