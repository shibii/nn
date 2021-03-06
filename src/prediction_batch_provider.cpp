#include "prediction_batch_provider.hpp"

namespace nn {
PredictionBatchProvider::PredictionBatchProvider(std::vector<float> sample_data,
                                                 DataShape sample_dim) {
  samples_ = af::array(sample_dim, sample_data.data());
}
PredictionBatchProvider::PredictionBatchProvider(af::array samples) {
  samples_ = samples;
}
PredictionBatch PredictionBatchProvider::batch(std::vector<long long> indices) {
  std::vector<float> floatindices(indices.begin(), indices.end());
  af::array index(floatindices.size(), floatindices.data());
  PredictionBatch batch(af::lookup(samples_, index, 3));
  return batch;
}
PredictionBatch PredictionBatchProvider::batch(long long from,
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