#pragma once

namespace nn {
struct Hyperparameters {
  Hyperparameters() : learningrate(1e-3f), weight_decay(0.f) {}
  float learningrate;
  float weight_decay;
  unsigned int batch_size;
};
}  // namespace nn