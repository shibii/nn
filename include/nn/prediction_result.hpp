#pragma once

#include <arrayfire.h>
#include <vector>

namespace nn {
class PredictionResult {
 public:
  PredictionResult(af::array output);
  unsigned int classify() const;
  std::vector<uint8_t> classify(float threshold) const;
  std::vector<float> output_raw() const;

 private:
  af::array output_;
};
}  // namespace nn