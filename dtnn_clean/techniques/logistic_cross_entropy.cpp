#include "logistic_cross_entropy.hpp"

namespace dtnn {
  af::array LogisticCrossEntropy::error(Feed &f, af::array target) const {
    return -(target - logistic(f.signal));
  }
  af::array LogisticCrossEntropy::loss(Feed &f, af::array target) const {
    af::array clamped = af::clamp(f.signal, 0.f, af::Inf);
    return clamped - f.signal * target + af::log(1.f + af::exp(-af::abs(f.signal)));
  }
  af::array LogisticCrossEntropy::output(Feed &f) const {
    return logistic(f.signal);
  }
  template <class Archive> void LogisticCrossEntropy::serialize(Archive &ar) {
  }
  af::array LogisticCrossEntropy::logistic(const af::array &input) const {
    return 1.f / (1.f + af::exp(-input));
  }
}