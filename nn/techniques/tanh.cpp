#include "tanh.hpp"

namespace nn {
void Tanh::forward(Feed &f) {
  activation_ = af::tanh(f.signal);
  f.signal = activation_;
}
void Tanh::backward(Feed &f) {
  f.signal = f.signal * (1.f - af::pow(activation_, 2));
}
template <class Archive>
void Tanh::serialize(Archive &ar) {}
}  // namespace nn