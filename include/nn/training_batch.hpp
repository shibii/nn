#pragma once

#include <arrayfire.h>
#include <vector>

#include "data_shape.hpp"

namespace nn {
class TrainingBatch {
 public:
  TrainingBatch(std::vector<float> sample_data, DataShape sample_dim,
                std::vector<float> target_data, DataShape target_dim);
  TrainingBatch(af::array samples, af::array targets);

 private:
  friend class Network;
  af::array samples_;
  af::array targets_;
};
}  // namespace nn