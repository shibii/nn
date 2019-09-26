#pragma once

#include <vector>
#include <arrayfire.h>

#include "prediction_batch.hpp"

namespace dtnn {
  class PredictionBatchProvider {
  public:
    PredictionBatchProvider(std::vector<float> sample_data, af::dim4 sampledim);
    PredictionBatchProvider(af::array samples);
    PredictionBatch batch(std::vector<float> indices);
    PredictionBatch batch(unsigned int from, unsigned int batch_size);
    unsigned int sample_count();
    af::dim4 sample_dimensions();
  private:
    af::array samples_;
  };
}