#pragma once

#include "serialization.hpp"
#include "wb.hpp"

namespace dtnn {
  struct OptimizableWeights {
    wb weights;
    wb gradient;
  };

  template<class Archive> void serialize(Archive & archive, OptimizableWeights & m)
  {
    archive(m.weights, m.gradient);
  }
}