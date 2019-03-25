#pragma once

#include <vector>
#include <arrayfire.h>

namespace dtnn {
  namespace util {
    static af::array column_batch(const af::array &a) {
      af::dim4 column(a.dims(0) * a.dims(1) * a.dims(2), 1, 1, a.dims(3));
      return af::moddims(a, column);
    }
    static void randomize_batches(af::array &input, af::array &target) {
      dim_t samples = input.dims(3);
      af::array randoms = af::randu(samples);
      af::array sorted, indices;
      af::sort(sorted, indices, randoms);
      input = input(af::span, af::span, af::span, indices);
      target = target(af::span, af::span, af::span, indices);
    }
    static af::array replace_zeroes(const af::array &a, float r = 1e-6f) {
      return a + (a == 0.f) * r;
    }
    static std::vector<float> vectorize(const af::array &a) {
      std::vector<float> vec(a.elements());
      a.host(vec.data());
      return vec;
    }
  }
}