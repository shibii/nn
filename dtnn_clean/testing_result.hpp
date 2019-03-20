#pragma once

#include <arrayfire.h>

namespace dtnn {
  class TestingResult {
  public:
    TestingResult(af::array output, af::array target, af::array loss);
    std::vector<float> output_raw();
    std::vector<float> target_raw();
    std::vector<float> loss_raw();
    float loss();
    float rmse();
    float accuracy();
    float accuracy(float threshold);

  private:
    af::array column_batch(af::array &a);

    af::array output_;
    af::array target_;
    af::array loss_;
  };
}