#pragma once

#include <arrayfire.h>
namespace dtnn {
  namespace test {
    bool samedim(af::array a, af::array b) {
      return a.dims() == b.dims();
    }
    bool approx(af::array a, af::array b, float sigma = 1e-3f) {
      af::array err = af::abs(a - b) > sigma;
      return af::sum<int>(err) == 0;
    }
    bool nonans(af::array a) {
      return af::sum<int>(af::isNaN(a)) == 0;
    }
    bool noinfs(af::array a) {
      return af::sum<int>(af::isInf(a)) == 0;
    }
    float isnumber(af::array a) {
      return noinfs(a) || nonans(a);
    }
  }
}