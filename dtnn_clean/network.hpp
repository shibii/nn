#pragma once

#include <memory>
#include <vector>
#include <arrayfire.h>

#include "sample_provider.hpp"
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
    Network(SampleProvider &provider, std::shared_ptr<Optimizer> optimizer);
    void add(std::shared_ptr<WeightlessStage> stage);
    void add(std::shared_ptr<WeightedStage> stage);
    void add(std::shared_ptr<LossFunction> loss);
    void train(TrainingProvider &provider, dim_t batchsize);
    void test(TrainingProvider &provider, dim_t batchsize);
    void predict(PredictionProvider &provider, dim_t batchsize);
  private:
    std::vector<std::shared_ptr<PropagationStage>> stages_;
    std::shared_ptr<LossFunction> loss_;
    std::shared_ptr<Optimizer> optimizer_;
    af::dim4 inputdim_;
  };
}