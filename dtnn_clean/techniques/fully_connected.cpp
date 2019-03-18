#include "fully_connected.hpp"

namespace dtnn {
  FullyConnected::FullyConnected(dim_t units) : units_(units)
  {
  }
  void FullyConnected::forward(Feed &f) {
    inputdim_ = f.signal.dims();
    // spatial dimensions and channels are flattened
    af::dim4 flatdim(inputdim_[0] * inputdim_[1] * inputdim_[2], inputdim_[3]);
    inputflat_ = af::moddims(f.signal, flatdim);
    // weighted connections are calculated
    af::array output = af::matmul(param_->weights.w, inputflat_) +
      af::tile(param_->weights.b, 1, flatdim[1]);
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
  std::shared_ptr<OptimizableWeights> FullyConnected::init(Feed sample) {
    af::dim4 indim = sample.signal.dims();
    dim_t elements = indim[0] * indim[1] * indim[2];

    auto w = wb(af::dim4(units_, elements), units_, 3.6 / sqrtf(units_));
    auto g = wb(af::dim4(units_, elements), units_);
    OptimizableWeights ow = { w, g };
    param_ = std::make_shared<OptimizableWeights>(ow);
    return param_;
  }
  template<class Archive> void FullyConnected::serialize(Archive & archive)
  {
    archive(param_, units_);
  }
}