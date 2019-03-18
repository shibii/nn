#pragma once

#include <arrayfire.h>

#include "prediction_provider.hpp"

namespace dtnn {
  class PredictionBlob : public PredictionProvider {
  public:
    PredictionBlob(float* inputs, af::dim4 inputdim);
    PredictionBatch& batch(dim_t size) override;
  private:
    PredictionBatch current_batch_;
    af::array inputs_;
    dim_t location_;
  };
}