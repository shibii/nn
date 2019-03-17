#include "prediction_blob.hpp"

namespace dtnn {
  PredictionBlob::PredictionBlob(float* inputs, af::dim4 inputdim) : location_(0) {
    inputs_ = af::array(inputdim, inputs);
  }
  PredictionBatch& PredictionBlob::batch(dim_t size) {
    if (location_ + size >= inputs_.dims(3))
      location_ = 0;
    current_batch_.inputs = inputs_(af::span, af::span, af::span, af::seq(location_, size - 1));
    return current_batch_;
  }
}