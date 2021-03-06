#include "softmax_cross_entropy.hpp"
#include "../arrayfire_util.hpp"

namespace nn {
af::array SoftmaxCrossEntropy::error(Feed &f, const af::array target) const {
  return softmax(f.signal) - target;
}
af::array SoftmaxCrossEntropy::loss(Feed &f, const af::array target) const {
  // shifted values are used to calculate the log of softmax
  af::array max = af::max(f.signal);
  af::array logexpsum =
      af::log(af::sum(af::exp(af::batchFunc(f.signal, max, util::sub))));
  af::array logsoftmax = af::batchFunc(f.signal, max, util::sub);
  logsoftmax = af::batchFunc(logsoftmax, logexpsum, util::sub);
  return -target * logsoftmax;
}
af::array SoftmaxCrossEntropy::output(Feed &f) const {
  return softmax(f.signal);
}
template <class Archive>
void SoftmaxCrossEntropy::serialize(Archive &ar) {}
af::array SoftmaxCrossEntropy::softmax(const af::array &input) const {
  // output values are shifted to avoid overflow
  af::array max = af::max(input);
  af::array exp = af::exp(af::batchFunc(input, max, util::sub));
  af::array expsum = af::sum(exp);
  return af::batchFunc(exp, expsum, util::div);
}
}  // namespace nn