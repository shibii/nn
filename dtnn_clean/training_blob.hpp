#pragma once

#include "training_provider.hpp"

namespace dtnn {
  class TrainingBlob : public TrainingProvider {
  public:
    TrainingBlob(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim);
    TrainingBatch& batch(dim_t size) override;
    TrainingBatch current_batch_;
  private:
    af::array inputs_;
    af::array targets_;
    dim_t location_;
  };
}