#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct PredictionBatch {
    af::array inputs;
  };

  class PredictionProvider {
  public:
    virtual PredictionBatch& batch(dim_t size) = 0;
    virtual af::dim4 input_dimensions() = 0;
  };
}