#include "logistic_cross_entropy.hpp"

namespace dtnn {
  af::array LogisticCrossEntropy::error(Feed &f, af::array target) {
    return -(target - logistic(f.signal));
  }
  af::array LogisticCrossEntropy::loss(Feed &f, af::array target) {
    af::array lhs = 1.f / (1.f + af::exp(-f.signal));
    af::array rhs = af::exp(-f.signal) / (1.f + af::exp(-f.signal));
    af::replace(rhs, !af::isNaN(rhs), 1.f);
    return target * lhs + !target * rhs;
  }
  af::array LogisticCrossEntropy::output(Feed &f) {
    return logistic(f.signal);
  }
  template <class Archive> void LogisticCrossEntropy::serialize(Archive &ar) {
  }
  af::array LogisticCrossEntropy::logistic(const af::array &input) {
    return 1.f / (1.f + af::exp(-input));
  }
}