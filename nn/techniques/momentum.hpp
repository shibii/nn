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
class Momentum : public Optimizer {
 public:
  ~Momentum() = default;
  Momentum(float decay = 0.1f);
  void optimize(Hyperparameters hp) override;
  void attach(std::shared_ptr<OptimizableWeights> param) override;
  template <class Archive>
  void serialize(Archive &ar);

 private:
  float decay_;
  struct OptimizerState;
  std::vector<OptimizerState> states_;
};

struct Momentum::OptimizerState {
  std::shared_ptr<OptimizableWeights> param;
  wb velocities;
  template <class Archive>
  void serialize(Archive &ar) {
    ar(param, velocities);
  }
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::Momentum);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::Optimizer, nn::Momentum)