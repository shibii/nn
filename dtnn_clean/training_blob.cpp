#include "training_blob.hpp"
#include "arrayfire_util.hpp"

namespace dtnn {
  TrainingBlob::TrainingBlob(af::array inputs, af::array targets)
    : inputs_(inputs), targets_(targets), location_(0)
  {
  }
  TrainingBlob::TrainingBlob(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim)
    : location_(0)
  {
    inputs_ = af::array(inputdim, inputs);
    targets_ = af::array(targetdim, targets);
  }
  const TrainingBatch& TrainingBlob::batch(dim_t size) {
    if (location_ + size > inputs_.dims(3)) location_ = 0;

    auto range = af::seq(location_, location_ + size - 1);
    current_batch_.inputs = inputs_(af::span, af::span, af::span, range);
    current_batch_.targets = targets_(af::span, af::span, af::span, range);
    location_ += size;
    return current_batch_;
  }
  af::dim4 TrainingBlob::input_dimensions() const {
    return af::dim4(inputs_.dims(0), inputs_.dims(1), inputs_.dims(2), 1);
  }
  dim_t TrainingBlob::samples() const {
    return inputs_.dims(3);
  }
}