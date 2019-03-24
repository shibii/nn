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
  class Momentum : public Optimizer {
  public:
    Momentum() = default;
    ~Momentum() = default;
    Momentum(float learningrate, float decay);
    void optimize() override;
    void attach(std::shared_ptr<OptimizableWeights> param) override;
    template <class Archive> void serialize(Archive &ar);

    float learningrate_;
    float decay_;
    struct OptimizerState;
    std::vector<OptimizerState> states_;
  };

  struct Momentum::OptimizerState {
    std::shared_ptr<OptimizableWeights> param;
    wb velocities;
    template <class Archive> void serialize(Archive &ar) { ar(param, velocities); }
  };
}
CEREAL_REGISTER_TYPE(dtnn::Momentum);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::Optimizer, dtnn::Momentum)