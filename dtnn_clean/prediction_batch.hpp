#pragma once

#include <arrayfire.h>
#include <vector>

namespace nn {
class PredictionBatch {
 public:
  PredictionBatch(std::vector<float> sample_data, af::dim4 sample_dim);
  PredictionBatch(af::array samples);

 private:
  friend class Network;
  af::array samples_;
};
}  // namespace nn