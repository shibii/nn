#pragma once

#include "wb.hpp"

namespace dtnn {
  struct OptimizableWeights {
    wb weights;
    wb gradient;

    template <class Archive> void serialize(Archive &ar) { ar(weights, gradient); }
  };
}