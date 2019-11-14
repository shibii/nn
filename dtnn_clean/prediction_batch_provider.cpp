#include "prediction_batch_provider.hpp"

namespace nn {
PredictionBatchProvider::PredictionBatchProvider(std::vector<float> sample_data,
                                                 af::dim4 sampledim) {
  samples_ = af::array(sampledim, sample_data.data());
}
PredictionBatchProvider::PredictionBatchProvider(af::array samples) {
  samples_ = samples;
}
PredictionBatch PredictionBatchProvider::batch(std::vector<float> indices) {
  af::array index(indices.size(), indices.data());
  PredictionBatch batch(af::lookup(samples_, index, 3));
  return batch;
}
PredictionBatch PredictionBatchProvider::batch(unsigned int from,
                                               unsigned int batch_size) {
  auto range = af::seq(from, from + batch_size - 1);
  auto batch_samples = samples_(af::span, af::span, af::span, range);
  return PredictionBatch(batch_samples);
}
unsigned int PredictionBatchProvider::sample_count() {
  return (unsigned int)samples_.dims(3);
}
af::dim4 PredictionBatchProvider::sample_dimensions() {
  return {samples_.dims(0), samples_.dims(1), samples_.dims(2)};
}
}  // namespace nn