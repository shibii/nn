#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct Batch {
    virtual af::array get_inputs() = 0;
  };

  class SampleProvider {
  public:
    virtual Batch& batch(dim_t size) = 0;
  };
}