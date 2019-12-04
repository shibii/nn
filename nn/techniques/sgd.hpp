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
class SGD : public Optimizer {
 public:
  SGD() = default;
  ~SGD() = default;
  void optimize(Hyperparameters hp) override;
  void attach(std::shared_ptr<OptimizableWeights> param) override;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive &ar);

  std::vector<std::shared_ptr<OptimizableWeights>> params_;
};
}  // namespace nn
CEREAL_REGISTER_TYPE(nn::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::Optimizer, nn::SGD)