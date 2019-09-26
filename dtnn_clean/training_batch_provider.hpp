#pragma once

#include <vector>
#include <arrayfire.h>

#include "training_batch.hpp"

namespace dtnn {
  class TrainingBatchProvider {
  public:
    TrainingBatchProvider(std::vector<float> sample_data, af::dim4 sampledim, std::vector<float> target_data, af::dim4 targetdim);
    TrainingBatchProvider(af::array samples, af::array targets);
    TrainingBatch batch(std::vector<float> indices);
    TrainingBatch batch(unsigned int from, unsigned int batch_size);
    unsigned int sample_count();
    af::dim4 sample_dimensions();
  private:
    af::array samples_;
    af::array targets_;
  };
}