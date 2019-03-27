#include "convolutional.hpp"
#include "../arrayfire_util.hpp"

namespace dtnn {
  Convolutional::Convolutional(dim_t size0, dim_t size1, dim_t stride0, dim_t stride1, dim_t pad0, dim_t pad1, dim_t features)
    : size0_(size0), size1_(size1), stride0_(stride0), stride1_(stride1), pad0_(pad0), pad1_(pad1), features_(features)
  {
  }
  void Convolutional::forward(Feed &f) {
    inputdim_ = f.signal.dims();
    // individual windows are laid into columns
    af::array unwrapped = af::unwrap(f.signal, size0_, size1_, stride0_, stride1_,
      pad0_, pad1_);
    // reordering the input into matrix in such way that
    // each feature of the corresponding window are in the same colums
    // and each window of each sample of the batch are in seperate rows
    af::array reorderinput = af::reorder(unwrapped, 0, 2, 1, 3);
    reorderdim_ = reorderinput.dims();
    af::dim4 flatdim(reorderinput.dims(0) * reorderinput.dims(1),
      reorderinput.dims(2) * reorderinput.dims(3));
    inputflat_ = af::moddims(reorderinput, flatdim);
    // performing convolution
    af::array convolution = af::matmulTN(inputflat_, param_->weights.w);
    convolution = af::batchFunc(convolution, param_->weights.b, util::add);

    // convolution result modified such that spatial,
    // channel and batch dimensions are correct
    convolutiondim_ = convolution.dims();
    dim_t windows0 = 1 + (inputdim_[0] + 2 * pad0_ - size0_) / stride0_;
    dim_t windows1 = 1 + (inputdim_[1] + 2 * pad1_ - size1_) / stride1_;
    af::dim4 outdim{ windows0 , windows1 , inputdim_[3] , features_ };
    convolution = af::moddims(convolution, outdim);
    f.signal = af::reorder(convolution, 0, 1, 3, 2);
  }
  void Convolutional::backward(Feed &f) {
    // walking backwards the forwards pass signal shaping steps
    af::array error = af::reorder(f.signal, 0, 1, 3, 2);
    error = af::moddims(error, convolutiondim_);
    // calculate the gradient
    param_->gradient.w += af::matmul(inputflat_, error);
    param_->gradient.b += af::sum(error, 0);
    // error is propagated wrt the weights
    error = af::matmulNT(param_->weights.w, error);
    // error is ordered and wrapped into the original forward propagated shape
    error = af::moddims(error, reorderdim_);
    error = af::reorder(error, 0, 2, 1, 3);
    f.signal = af::wrap(error, inputdim_[0], inputdim_[1], size0_, size1_, stride0_, stride1_, pad0_, pad1_);
  }
  std::shared_ptr<OptimizableWeights> Convolutional::init(af::dim4 input) {
    af::dim4 kerneldim = {size0_ * size1_ * input[2], features_};
    auto w = wb(kerneldim, af::dim4(1, features_), sqrtf(2.f / (float)(size0_ * size1_ * input[2])));
    auto g = wb(kerneldim, af::dim4(1, features_));
    OptimizableWeights ow = { w, g };
    param_ = std::make_shared<OptimizableWeights>(ow);
    return param_;
  }
  template <class Archive> void Convolutional::serialize(Archive &ar) {
    ar(param_, size0_, size1_, stride0_, stride1_, pad0_, pad1_, features_);
  }
}