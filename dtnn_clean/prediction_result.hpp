#pragma once

#include <vector>
#include <arrayfire.h>

namespace dtnn {
  class PredictionResult {
  public:
    PredictionResult(af::array output);
    unsigned int classify();
    std::vector<uint8_t> classify(float threshold);
    std::vector<float> output_raw();

  private:
    af::array column_batch(af::array &a);

    af::array output_;
  };
}