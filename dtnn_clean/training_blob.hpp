#pragma once

#include <arrayfire.h>

#include "training_provider.hpp"

namespace dtnn {
  class TrainingBlob : public TrainingProvider {
  public:
    TrainingBlob(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim);
    TrainingBatch& batch(dim_t size) override;
    af::dim4 input_dimensions() override;
  private:
    TrainingBatch current_batch_;
    af::array inputs_;
    af::array targets_;
    dim_t location_;
  };
}