#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct Samples {
    af::array inputs;
    af::array targets;
  };

  class SampleProvider {
  public:
    virtual Samples batch(dim_t size) = 0;
  };
}