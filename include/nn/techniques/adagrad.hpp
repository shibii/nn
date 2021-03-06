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
class Adagrad : public Optimizer {
 public:
  Adagrad() = default;
  ~Adagrad() = default;
  void optimize(Hyperparameters hp) override;
  void attach(std::shared_ptr<OptimizableWeights> param) override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);

  struct OptimizerState;
  std::vector<OptimizerState> states_;
};

struct Adagrad::OptimizerState {
  std::shared_ptr<OptimizableWeights> param;
  wb sum_of_squared_grad;
  template <class Archive>
  void serialize(Archive &ar) {
    ar(param, sum_of_squared_grad);
  }
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::Adagrad);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::Optimizer, nn::Adagrad)