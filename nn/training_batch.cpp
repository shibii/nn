#include "training_batch.hpp"

namespace nn {
TrainingBatch::TrainingBatch(std::vector<float> sample_data,
                             DataShape sample_dim,
                             std::vector<float> target_data,
                             DataShape target_dim) {
  samples_ = af::array(sample_dim, sample_data.data());
  targets_ = af::array(target_dim, target_data.data());
}
TrainingBatch::TrainingBatch(af::array samples, af::array targets) {
  samples_ = samples;
  targets_ = targets;
}
}  // namespace nn