#pragma once

#include <memory>
#include <vector>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#include "../hyperparameters.hpp"
#include "../cereal_archives.hpp"
#include "../optimizer.hpp"
#include "../optimizable_weights.hpp"

namespace nn {
  class Adam : public Optimizer {
  public:
    ~Adam() = default;
    Adam(float decay1 = 0.9f, float decay2 = 0.999f);
    void optimize(Hyperparameters hp) override;
    void attach(std::shared_ptr<OptimizableWeights> param) override;
    template <class Archive> void serialize(Archive &ar);

    float decay1_;
    float decay2_;
    float decay1T_;
    float decay2T_;
    struct OptimizerState;
    std::vector<OptimizerState> states_;
  };

  struct Adam::OptimizerState {
    std::shared_ptr<OptimizableWeights> param;
    wb m;
    wb v;
    template <class Archive> void serialize(Archive &ar) { ar(param, m, v); }
  };
}
CEREAL_REGISTER_TYPE(nn::Adam);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::Optimizer, nn::Adam)