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
  class RMSprop : public Optimizer {
  public:
    RMSprop() = default;
    ~RMSprop() = default;
    RMSprop(float learningrate, float decay = 0.1f);
    void optimize() override;
    void attach(std::shared_ptr<OptimizableWeights> param) override;
    template <class Archive> void serialize(Archive &ar);

    float learningrate_;
    float decay_;
    struct OptimizerState;
    std::vector<OptimizerState> states_;
  };

  struct RMSprop::OptimizerState {
    std::shared_ptr<OptimizableWeights> param;
    wb rms;
    template <class Archive> void serialize(Archive &ar) { ar(param, rms); }
  };
}
CEREAL_REGISTER_TYPE(dtnn::RMSprop);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::Optimizer, dtnn::RMSprop)