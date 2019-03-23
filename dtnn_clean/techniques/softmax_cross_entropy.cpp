#include "softmax_cross_entropy.hpp"

namespace dtnn {
  af::array SoftmaxCrossEntropy::error(Feed &f, af::array target) {
    return -(target - softmax(f.signal));
  }
  af::array SoftmaxCrossEntropy::loss(Feed &f, af::array target) {
    // the shifted values are also used to calculate the log of softmax
    af::array max = af::tile(af::max(f.signal), (unsigned int)f.signal.dims(0));
    af::array exp = af::exp(f.signal - max);
    af::array expsum = af::tile(af::sum(exp), (unsigned int)f.signal.dims(0));
    af::array logsoftmax = f.signal - (max + af::log(expsum));
    return -(target * logsoftmax);
  }
  af::array SoftmaxCrossEntropy::output(Feed &f) {
    return softmax(f.signal);
  }
  template <class Archive> void SoftmaxCrossEntropy::serialize(Archive &ar) {
  }
  af::array SoftmaxCrossEntropy::softmax(const af::array &input) {
    // output values are shifted to avoid overflow
    af::array max = af::tile(af::max(input), (unsigned int)input.dims(0));
    af::array exp = af::exp(input - max);
    af::array expsum = af::tile(af::sum(exp), (unsigned int)input.dims(0));
    return exp / expsum;
  }
}