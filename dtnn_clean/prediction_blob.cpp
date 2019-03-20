#include "prediction_blob.hpp"

namespace dtnn {
  PredictionBlob::PredictionBlob(float* inputs, af::dim4 inputdim) : location_(0) {
    inputs_ = af::array(inputdim, inputs);
  }
  PredictionBatch& PredictionBlob::batch(dim_t size) {
    if (location_ + size >= inputs_.dims(3))
      location_ = 0;
    current_batch_.inputs = inputs_(af::span, af::span, af::span, af::seq((double)location_, (double)size - 1));
    return current_batch_;
  }
  af::dim4 PredictionBlob::input_dimensions() {
    return { inputs_.dims(0), inputs_.dims(1), inputs_.dims(2) };
  }
}