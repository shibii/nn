#pragma once

#include <arrayfire.h>

#include "sample_provider.hpp"

namespace dtnn {
  struct TrainingBatch : public Batch {
    af::array inputs;
    af::array targets;
  };

  class TrainingProvider : public SampleProvider {
  public:
    virtual TrainingBatch& batch(dim_t size) = 0;
  };
}