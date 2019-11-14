#pragma once

#include <arrayfire.h>

#include "feed.hpp"

namespace nn {
class LossFunction {
 public:
  LossFunction() = default;
  virtual ~LossFunction() = default;
  virtual af::array error(Feed &f, af::array target) const = 0;
  virtual af::array loss(Feed &f, af::array target) const = 0;
  virtual af::array output(Feed &f) const = 0;
};
}  // namespace nn