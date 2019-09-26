#include "training_batch.hpp"

namespace dtnn
{
  TrainingBatch::TrainingBatch(std::vector<float> sample_data, std::vector<float> target_data, af::dim4 sample_dim, af::dim4 target_dim) {
    samples_ = af::array(sample_dim, sample_data.data());
    targets_ = af::array(target_dim, target_data.data());
  }
  TrainingBatch::TrainingBatch(af::array samples, af::array targets) {
    samples_ = samples;
    targets_ = targets;
  }
}