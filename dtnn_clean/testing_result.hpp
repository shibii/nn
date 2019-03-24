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
    float precision(float threshold);
    float recall(float threshold);
    float f1(float threshold);
    float specificity(float threshold);
    float accuracy(float threshold);
    float true_positive(float threshold);
    float true_negative(float threshold);
    float false_positive(float threshold);
    float false_negative(float threshold);
    float accuracy();
  private:
    af::array output_;
    af::array target_;
    af::array loss_;

  };
}