#include "squared_loss.hpp"

namespace nn {
af::array SquaredLoss::error(Feed &f, const af::array target) const {
  return -(target - f.signal);
}
af::array SquaredLoss::loss(Feed &f, const af::array target) const {
  return .5f * af::pow(target - f.signal, 2);
}
af::array SquaredLoss::output(Feed &f) const { return f.signal; }
template <class Archive>
void SquaredLoss::serialize(Archive &ar) {}
}  // namespace nn