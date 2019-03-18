#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct Batch {
  };

  class SampleProvider {
  public:
    virtual Batch& batch(dim_t size) = 0;
    virtual af::dim4 input_dimensions() = 0;
  };
}