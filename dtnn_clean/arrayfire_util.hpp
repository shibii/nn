#pragma once

#include <vector>
#include <arrayfire.h>

namespace dtnn {
  namespace util {
    static af::array column_batch(const af::array &a) {
      af::dim4 column(a.dims(0) * a.dims(1) * a.dims(2), 1, 1, a.dims(3));
      return af::moddims(a, column);
    }
    static std::vector<float> vectorize(const af::array &a) {
      std::vector<float> vec(a.elements());
      a.host(vec.data());
      return vec;
    }
  }
}