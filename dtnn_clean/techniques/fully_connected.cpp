#include "fully_connected.hpp"
#include "../arrayfire_util.hpp"

namespace nn {
FullyConnected::FullyConnected(dim_t units) : units_(units) {}
void FullyConnected::forward(Feed &f) {
  inputdim_ = f.signal.dims();
  // spatial dimensions and channels are flattened
  af::dim4 flatdim(inputdim_[0] * inputdim_[1] * inputdim_[2], inputdim_[3]);
  inputflat_ = af::moddims(f.signal, flatdim);
  // weighted connections are calculated
  af::array output = af::matmul(param_->weights.w, inputflat_);
  output = af::batchFunc(output, param_->weights.b, util::add);
  // batch dimension is restored
  f.signal = af::moddims(output, output.dims(0), 1, 1, output.dims(1));
}
void FullyConnected::backward(Feed &f) {
  // modifying the error from tensor form into matrix form
  af::dim4 flatdim(f.signal.dims(0) * f.signal.dims(1) * f.signal.dims(2),
                   f.signal.dims(3));
  af::array errorflat = af::moddims(f.signal, flatdim);
  // calculating the gradient
  param_->gradient.w += af::matmulNT(errorflat, inputflat_);
  param_->gradient.b += af::sum(errorflat, 1);
  // calculating the new backpropagated errors
  af::array output = af::matmulTN(param_->weights.w, errorflat);
  // error shape is modified to match the input received in forward pass
  f.signal = af::moddims(output, inputdim_);
}
std::shared_ptr<OptimizableWeights> FullyConnected::init(af::dim4 input) {
  dim_t elements = input[0] * input[1] * input[2];
  float limit = sqrtf(2.f / (float)elements);
  auto w = wb(af::dim4(units_, elements), af::dim4(units_), limit);
  auto g = wb(af::dim4(units_, elements), af::dim4(units_));
  OptimizableWeights ow = {w, g};
  param_ = std::make_shared<OptimizableWeights>(ow);
  return param_;
}
template <class Archive>
void FullyConnected::serialize(Archive &ar) {
  ar(param_, units_);
}
}  // namespace nn