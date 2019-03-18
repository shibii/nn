#pragma once

#include <memory>
#include <vector>
#include <arrayfire.h>

#include "training_provider.hpp"
#include "prediction_provider.hpp"

#include "optimizer.hpp"

#include "propagation_stage.hpp"
#include "weighted_stage.hpp"
#include "weightless_stage.hpp"

#include "loss_function.hpp"

namespace dtnn {
  class Network {
  public:
    Network(af::dim4 input_dimensions, std::shared_ptr<Optimizer> optimizer);
    void add(std::shared_ptr<WeightlessStage> stage);
    void add(std::shared_ptr<WeightedStage> stage);
    void add(std::shared_ptr<LossFunction> loss);
    void train(TrainingBatch &batch);
    af::array test(TrainingBatch &batch);
    af::array predict(PredictionBatch &batch);
  private:
    std::vector<std::shared_ptr<PropagationStage>> stages_;
    std::shared_ptr<LossFunction> loss_;
    std::shared_ptr<Optimizer> optimizer_;
    af::dim4 inputdim_;
  };
}