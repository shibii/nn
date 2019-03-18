#pragma once

#include <arrayfire.h>

#include "sample_provider.hpp"

namespace dtnn {
  struct PredictionBatch : public Batch {
    af::array inputs;
  };

  class PredictionProvider : public SampleProvider {
  public:
    virtual PredictionBatch& batch(dim_t size) = 0;
  };
}