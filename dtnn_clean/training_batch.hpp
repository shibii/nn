#pragma once

#include <vector>
#include <arrayfire.h>

namespace nn {
  class TrainingBatch {
  public:
    TrainingBatch(std::vector<float> sample_data, std::vector<float> target_data, af::dim4 sample_dim, af::dim4 target_dim);
    TrainingBatch(af::array samples, af::array targets);
  private:
    friend class Network;
    af::array samples_;
    af::array targets_;
  };
}