#pragma once

#include <memory>
#include <vector>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include "../cereal_archives.hpp"
#include "../optimizer.hpp"
#include "../optimizable_weights.hpp"

namespace dtnn {
  class Adagrad : public Optimizer {
  public:
    Adagrad() = default;
    ~Adagrad() = default;
    Adagrad(float learningrate);
    void optimize(unsigned int batch_size) override;
    void attach(std::shared_ptr<OptimizableWeights> param) override;
    template <class Archive> void serialize(Archive &ar);

    float learningrate_;
    struct OptimizerState;
    std::vector<OptimizerState> states_;
  };

  struct Adagrad::OptimizerState {
    std::shared_ptr<OptimizableWeights> param;
    wb sum_of_squared_grad;
    template <class Archive> void serialize(Archive &ar) { ar(param, sum_of_squared_grad); }
  };
}
CEREAL_REGISTER_TYPE(dtnn::Adagrad);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::Optimizer, dtnn::Adagrad)