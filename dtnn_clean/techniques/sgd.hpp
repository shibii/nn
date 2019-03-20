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
  class SGD : public Optimizer {
  public:
    SGD() = default;
    ~SGD() = default;
    SGD(float learningrate);
    void optimize() override;
    void attach(std::shared_ptr<OptimizableWeights> param) override;
    template <class Archive> void serialize(Archive &ar);

    float learningrate_;
    std::vector<std::shared_ptr<OptimizableWeights>> params_;
  };
}
CEREAL_REGISTER_TYPE(dtnn::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::Optimizer, dtnn::SGD)