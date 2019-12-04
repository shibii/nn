#pragma once

#include <arrayfire.h>
#include <vector>

#include "data_shape.hpp"

namespace nn {
class PredictionBatch {
 public:
  PredictionBatch(std::vector<float> sample_data, DataShape sample_dim);
  PredictionBatch(af::array samples);

 private:
  friend class Network;
  af::array samples_;
};
}  // namespace nn