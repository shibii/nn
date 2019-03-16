#pragma once

#include "wb.hpp"

namespace dtnn {
  struct OptimizableWeights {
    wb weights;
    wb gradient;
  };
}