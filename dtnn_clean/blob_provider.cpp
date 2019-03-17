#include "blob_provider.hpp"

namespace dtnn {
  BlobProvider::BlobProvider(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim) {
    inputs_ = af::array(inputdim, inputs);
    targets_ = af::array(targetdim, targets);
  }
  Samples BlobProvider::batch(dim_t size) {
    if (location_ + size >= inputs_.dims(3))
      location_ = 0;
    Samples batch;
    batch.inputs = inputs_(af::span, af::span, af::span, af::seq(location_, size-1));
    batch.targets = targets_(af::span, af::span, af::span, af::seq(location_, size-1));
    return batch;
  }
}