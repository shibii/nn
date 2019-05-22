#pragma once

#include <memory>
#include <vector>
#include <arrayfire.h>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>

#include "arrayfire_serialization.hpp"
#include "training_provider.hpp"
#include "prediction_provider.hpp"
#include "optimizer.hpp"
#include "propagation_stage.hpp"
#include "weighted_stage.hpp"
#include "weightless_stage.hpp"
#include "loss_function.hpp"
#include "testing_result.hpp"
#include "prediction_result.hpp"
#include "hyperparameters.hpp"

namespace dtnn {
  class Network {
  public:
    Network() = default;
    Network(af::dim4 input_dimensions, std::shared_ptr<Optimizer> optimizer);
    void add(std::shared_ptr<WeightlessStage> stage);
    void add(std::shared_ptr<WeightedStage> stage);
    void add(std::shared_ptr<LossFunction> loss);
    void generate_gradient(TrainingBatch &batch);
    void update_weights(Hyperparameters hyperparameters);
    void train(TrainingBatch &batch, Hyperparameters hyperparameters);
    TestingResult test(TrainingBatch &batch);
    PredictionResult predict(PredictionBatch &batch);
    template <class Archive> void serialize(Archive &ar) {
      ar(stages_, loss_, optimizer_, inputdim_);
    }

  private:
    void forward_stages(Feed &feed);
    void backward_stages(Feed &feed);

    std::vector<std::shared_ptr<PropagationStage>> stages_;
    std::shared_ptr<LossFunction> loss_;
    std::shared_ptr<Optimizer> optimizer_;
    af::dim4 inputdim_;
    unsigned int batch_samples_;
  };
}