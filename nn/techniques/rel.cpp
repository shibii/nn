#include "rel.hpp"

namespace nn {
void ReL::forward(Feed &f) {
  input_ = f.signal;
  f.signal = (input_ > 0.f) * input_;
}
void ReL::backward(Feed &f) { f.signal = (input_ > 0.f) * f.signal; }
template <class Archive>
void ReL::serialize(Archive &ar) {}
}  // namespace nn