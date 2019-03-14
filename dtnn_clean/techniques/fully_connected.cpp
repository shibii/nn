#include "fully_connected.hpp"

namespace dtnn {
  FullyConnected::FullyConnected(std::shared_ptr<wb> weights)
    : weights_(weights)
  {}
  void FullyConnected::forward(Pack &p) {
    inputdim_ = p.signal.dims();
    // spatial dimensions and channels are flattened
    af::dim4 flatdim(inputdim_[0] * inputdim_[1] * inputdim_[2], inputdim_[3]);
    inputflat_ = af::moddims(p.signal, flatdim);
    // weighted connections are calculated
    af::array output = af::matmul(weights_->w, inputflat_) +
      af::tile(weights_->b, 1, flatdim[1]);
    // batch dimension is restored
    p.signal = af::moddims(output, output.dims(0), 1, 1, output.dims(1));
  }
  void FullyConnected::backward(Pack &p) {

  }
}