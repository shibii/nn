#include "squared_loss.hpp"

namespace dtnn {
  af::array SquaredLoss::error(Feed &f, af::array target) const {
    return -(target - f.signal);
  }
  af::array SquaredLoss::loss(Feed &f, af::array target) const {
    return 0.5 * af::pow(target - f.signal, 2);
  }
  af::array SquaredLoss::output(Feed &f) const {
    return f.signal;
  }
  template <class Archive> void SquaredLoss::serialize(Archive &ar) {
  }
}