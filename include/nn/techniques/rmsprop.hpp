#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <vector>

#include "../cereal_archives.hpp"
#include "../hyperparameters.hpp"
#include "../optimizable_weights.hpp"
#include "../optimizer.hpp"

namespace nn {
class RMSprop : public Optimizer {
 public:
  ~RMSprop() = default;
  RMSprop(float decay = 0.1f);
  void optimize(Hyperparameters hp) override;
  void attach(std::shared_ptr<OptimizableWeights> param) override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);

  float decay_;
  struct OptimizerState;
  std::vector<OptimizerState> states_;
};

struct RMSprop::OptimizerState {
  std::shared_ptr<OptimizableWeights> param;
  wb rms;
  template <class Archive>
  void serialize(Archive &ar) {
    ar(param, rms);
  }
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::RMSprop);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::Optimizer, nn::RMSprop)