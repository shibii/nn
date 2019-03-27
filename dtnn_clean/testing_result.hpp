#pragma once

#include <arrayfire.h>

namespace dtnn {
  class TestingResult {
  public:
    TestingResult(af::array output, af::array target, af::array loss);
    std::vector<float> output_raw() const;
    std::vector<float> target_raw() const;
    std::vector<float> loss_raw() const;
    float loss() const;
    float rmse() const;
    float precision(float threshold) const;
    float recall(float threshold) const;
    float f1(float threshold) const;
    float specificity(float threshold) const;
    float accuracy(float threshold) const;
    float true_positive(float threshold) const;
    float true_negative(float threshold) const;
    float false_positive(float threshold) const;
    float false_negative(float threshold) const;
    float accuracy() const;
  private:
    af::array output_;
    af::array target_;
    af::array loss_;

  };
}