#pragma once

#include <vector>
#include <arrayfire.h>

namespace dtnn {
  class PredictionResult {
  public:
    PredictionResult(af::array output);
    std::vector<float> output_raw();

  private:
    af::array column_batch(af::array &a);

    af::array output_;
  };
}