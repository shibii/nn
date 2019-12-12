#pragma once

#include <arrayfire.h>
#include <vector>

#include "training_batch.hpp"
#include "data_shape.hpp"

namespace nn {
class TrainingBatchProvider {
 public:
  TrainingBatchProvider(std::vector<float> sample_data, DataShape sample_dim,
                        std::vector<float> target_data, DataShape target_dim);
  TrainingBatchProvider(af::array samples, af::array targets);
  TrainingBatch batch(std::vector<long long> indices);
  TrainingBatch batch(unsigned int from, unsigned int batch_size);
  unsigned int sample_count();
  af::dim4 sample_dimensions();

 private:
  af::array samples_;
  af::array targets_;
};
}  // namespace nn