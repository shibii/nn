#include "mean_pool.hpp"
#include "../arrayfire_util.hpp"

namespace nn {
MeanPool::MeanPool(dim_t size0, dim_t size1, dim_t stride0, dim_t stride1,
                   dim_t pad0, dim_t pad1)
    : size0_(size0),
      size1_(size1),
      stride0_(stride0),
      stride1_(stride1),
      pad0_(pad0),
      pad1_(pad1) {}
void MeanPool::forward(Feed &f) {
  inputdim_ = f.signal.dims();
  // individual windows are laid into columns
  af::array unwrapped =
      af::unwrap(f.signal, size0_, size1_, stride0_, stride1_, pad0_, pad1_);
  unwrapdim_ = unwrapped.dims();
  // claculate the mean value of each individual subsampling window
  af::array mean = af::mean(unwrapped, 0);
  // return spatial dimensions
  dim_t windows0 = 1 + (inputdim_[0] + 2 * pad0_ - size0_) / stride0_;
  dim_t windows1 = 1 + (inputdim_[1] + 2 * pad1_ - size1_) / stride1_;
  f.signal = af::moddims(mean, windows0, windows1, inputdim_[2], inputdim_[3]);
}
void MeanPool::backward(Feed &f) {
  af::array error = f.signal;
  // error laid out into rows
  error = af::moddims(error, 1, error.dims(0) * error.dims(1), error.dims(2),
                      error.dims(3));
  // error is equally distributed among elements within subsampling windows
  error = af::tile(error, (unsigned int)unwrapdim_[0]) / (size0_ * size1_);
  // error is wrapped back to the shape originally received in forward pass
  f.signal = af::wrap(error, inputdim_[0], inputdim_[1], size0_, size1_,
                      stride0_, stride1_, pad0_, pad1_);
}
template <class Archive>
void MeanPool::serialize(Archive &ar) {
  ar(size0_, size1_, stride0_, stride1_, pad0_, pad1_);
}
}  // namespace nn