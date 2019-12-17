#pragma once

#include <arrayfire.h>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <vector>

#include "arrayfire_serialization.hpp"
#include "hyperparameters.hpp"
#include "loss_function.hpp"
#include "optimizer.hpp"
#include "prediction_batch.hpp"
#include "prediction_result.hpp"
#include "propagation_stage.hpp"
#include "testing_result.hpp"
#include "training_batch.hpp"
#include "weighted_stage.hpp"
#include "weightless_stage.hpp"

namespace nn {
class Network {
 public:
  Network() = default;
  Network(af::dim4 input_dimensions, std::shared_ptr<Optimizer> optimizer);
  void add(std::shared_ptr<WeightlessStage> stage);
  void add(std::shared_ptr<WeightedStage> stage);
  void add(std::shared_ptr<LossFunction> loss);
  void generate_gradient(const TrainingBatch &batch);
  void update_weights(Hyperparameters hyperparameters);
  void train(const TrainingBatch &batch, Hyperparameters hyperparameters);
  TestingResult test(const TrainingBatch &batch);
  PredictionResult predict(const PredictionBatch &batch);
  void merge_weights(Network &from, float bias);
  std::vector<std::shared_ptr<OptimizableWeights>> get_weights();
  template <class Archive>
  void serialize(Archive &ar) {
    ar(stages_, loss_, optimizer_, inputdim_);
  }

 private:
  void forward_stages(Feed &feed);
  void backward_stages(Feed &feed);

  std::vector<std::shared_ptr<PropagationStage>> stages_;
  std::vector<std::shared_ptr<OptimizableWeights>> weights_;
  std::shared_ptr<LossFunction> loss_;
  std::shared_ptr<Optimizer> optimizer_;
  af::dim4 inputdim_;
  unsigned int batch_samples_;
};
}  // namespace nn