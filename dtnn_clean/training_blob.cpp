#include "training_blob.hpp"

namespace dtnn {
  TrainingBlob::TrainingBlob(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim) : location_(0) {
    inputs_ = af::array(inputdim, inputs);
    targets_ = af::array(targetdim, targets);
  }
  TrainingBatch& TrainingBlob::batch(dim_t size) {
    if (location_ + size >= inputs_.dims(3))
      location_ = 0;
    current_batch_.inputs = inputs_(af::span, af::span, af::span, af::seq((double)location_, (double)size-1));
    current_batch_.targets = targets_(af::span, af::span, af::span, af::seq((double)location_, (double)size-1));
    return current_batch_;
  }
  af::dim4 TrainingBlob::input_dimensions() {
    return { inputs_.dims(0), inputs_.dims(1), inputs_.dims(2) };
  }
}