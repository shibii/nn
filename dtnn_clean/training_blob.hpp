#pragma once

#include <arrayfire.h>

#include "training_provider.hpp"

namespace dtnn {
  class TrainingBlob : public TrainingProvider {
  public:
    TrainingBlob(af::array inputs, af::array targets);
    TrainingBlob(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim);
    const TrainingBatch& batch(dim_t size) override;
    af::dim4 input_dimensions() const override;
    dim_t samples() const;
  private:
    TrainingBatch current_batch_;
    af::array inputs_;
    af::array targets_;
    dim_t location_;
  };
}