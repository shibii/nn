#include "fully_connected.hpp"

namespace dtnn {
  FullyConnected::FullyConnected(std::shared_ptr<wb> weights, std::shared_ptr<wb> gradient)
    : weights_(weights), gradient_(gradient)
  {
  }
  FullyConnected::FullyConnected(std::shared_ptr<wb> weights)
    : weights_(weights)
  {
    gradient_ = std::make_shared<wb>(wb());
    gradient_->w = af::constant(0.f, weights_->w.dims());
    gradient_->b = af::constant(0.f, weights_->b.dims());
  }
  void FullyConnected::forward(Feed &f) {
    inputdim_ = f.signal.dims();
    // spatial dimensions and channels are flattened
    af::dim4 flatdim(inputdim_[0] * inputdim_[1] * inputdim_[2], inputdim_[3]);
    inputflat_ = af::moddims(f.signal, flatdim);
    // weighted connections are calculated
    af::array output = af::matmul(weights_->w, inputflat_) +
      af::tile(weights_->b, 1, flatdim[1]);
    // batch dimension is restored
    f.signal = af::moddims(output, output.dims(0), 1, 1, output.dims(1));
  }
  void FullyConnected::backward(Feed &f) {
    // modifying the error from tensor form into matrix form
    af::dim4 flatdim(f.signal.dims(0) * f.signal.dims(1) * f.signal.dims(2),
      f.signal.dims(3));
    af::array errorflat = af::moddims(f.signal, flatdim);
    // calculating the gradient
    gradient_->w += af::matmulNT(errorflat, inputflat_);
    gradient_->b += af::sum(errorflat, 1);
    // calculating the new backpropagated errors
    af::array output = af::matmulTN(weights_->w, errorflat);
    // error shape is modified to match the input received in forward pass
    f.signal = af::moddims(output, inputdim_);
  }
}