#pragma once

#include <arrayfire.h>

namespace dtnn {
  class TestingResult {
  public:
    TestingResult(af::array output, af::array target, af::array loss);
    float loss();
    float rmse();

  private:
    af::array column_batch(af::array &a);

    af::array output_;
    af::array target_;
    af::array loss_;
  };
}