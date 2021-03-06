#include "training_batch_provider.hpp"

namespace nn {
TrainingBatchProvider::TrainingBatchProvider(std::vector<float> sample_data,
                                             DataShape sample_dim,
                                             std::vector<float> target_data,
                                             DataShape target_dim) {
  samples_ = af::array(sample_dim, sample_data.data());
  targets_ = af::array(target_dim, target_data.data());
}
TrainingBatchProvider::TrainingBatchProvider(af::array samples,
                                             af::array targets) {
  samples_ = samples;
  targets_ = targets;
}
TrainingBatch TrainingBatchProvider::batch(std::vector<long long> indices) {
  std::vector<float> floatindices(indices.begin(), indices.end());
  af::array index(floatindices.size(), floatindices.data());
  TrainingBatch batch(af::lookup(samples_, index, 3),
                      af::lookup(targets_, index, 3));
  return batch;
}
TrainingBatch TrainingBatchProvider::batch(long long from,
                                           unsigned int batch_size) {
  auto range = af::seq(from, from + batch_size - 1);
  auto batch_samples = samples_(af::span, af::span, af::span, range);
  auto batch_targets = targets_(af::span, af::span, af::span, range);
  return TrainingBatch(batch_samples, batch_targets);
}
unsigned int TrainingBatchProvider::sample_count() {
  return (unsigned int)samples_.dims(3);
}
af::dim4 TrainingBatchProvider::sample_dimensions() {
  return {samples_.dims(0), samples_.dims(1), samples_.dims(2)};
}
}  // namespace nn