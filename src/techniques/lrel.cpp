#include "lrel.hpp"

namespace nn {
LReL::LReL(float leak) : leak_(leak) {}
void LReL::forward(Feed &f) {
  input_ = f.signal;
  f.signal = (input_ > 0.f) * input_ + (input_ <= 0.f) * input_ * leak_;
}
void LReL::backward(Feed &f) {
  f.signal = (input_ > 0.f) * f.signal + (input_ <= 0.f) * f.signal * leak_;
}
template <class Archive>
void LReL::serialize(Archive &ar) {
  ar(leak_);
}
}  // namespace nn