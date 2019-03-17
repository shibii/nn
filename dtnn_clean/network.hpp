#pragma once

#include "propagation_stage.hpp"
#include "loss_function.hpp"
#include "weighted_stage.hpp"
#include "weightless_stage.hpp"
#include "optimizer.hpp"
#include "sample_provider.hpp"

namespace dtnn {
  class Network {
  public:
    Network(std::shared_ptr<SampleProvider> provider);
    void add(std::shared_ptr<WeightlessStage> stage);
    void add(std::shared_ptr<WeightedStage> stage);
    void add(std::shared_ptr<LossFunction> loss);
    void add(std::shared_ptr<Optimizer> optimizer);
    void train(dim_t batchsize);
    void predict(dim_t batchsize);
    Samples currentbatch_;
  private:
    std::shared_ptr<SampleProvider> provider_;
    std::vector<std::shared_ptr<PropagationStage>> stages_;
    std::shared_ptr<LossFunction> loss_;
    std::shared_ptr<Optimizer> optimizer_;
  };
}