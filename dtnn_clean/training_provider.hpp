#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct TrainingBatch {
    af::array inputs;
    af::array targets;
  };

  class TrainingProvider {
  public:
    virtual TrainingBatch& batch(dim_t size) = 0;
    virtual af::dim4 input_dimensions() = 0;
  };
}