#pragma once

#include "../optimizable_weights.hpp";

namespace cereal {
  template<class Archive>
  void serialize(Archive & archive, dtnn::OptimizableWeights & m)
  {
    archive(m);
  }
}