#include "prediction_batch.hpp"

namespace dtnn
{
  PredictionBatch::PredictionBatch(std::vector<float> sample_data, af::dim4 sample_dim) {
    samples_ = af::array(sample_dim, sample_data.data());
  }
  PredictionBatch::PredictionBatch(af::array samples) {
    samples_ = samples;
  }
}