#include "dropout.hpp"

namespace dtnn {
  Dropout::Dropout(float pass_probability)
    : pass_probability_(pass_probability)
  {
  }
  void Dropout::forward(Feed &f) {
    passmask_ = af::randu(f.signal.dims()) < pass_probability_;
    f.signal = f.signal * (1.f / pass_probability_) * passmask_;
  }
  void Dropout::backward(Feed &f) {
    f.signal = f.signal * (1.f / pass_probability_) * passmask_;
  }
  template <class Archive> void Dropout::serialize(Archive &ar) {
    ar(pass_probability_);
  }
}