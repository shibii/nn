#include "dropout.hpp"

namespace nn {
  Dropout::Dropout(float pass_probability)
    : pass_probability_(pass_probability)
  {
  }
  void Dropout::forward(Feed &f) {
    float pb = f.is_training ? pass_probability_ : 1.f;
    passmask_ = af::randu(f.signal.dims()) <= pb;
    f.signal = f.signal * (1.f / pb) * passmask_;
  }
  void Dropout::backward(Feed &f) {
    f.signal = f.signal * (1.f / pass_probability_) * passmask_;
  }
  template <class Archive> void Dropout::serialize(Archive &ar) {
    ar(pass_probability_);
  }
}