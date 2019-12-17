#pragma once

#include <arrayfire.h>

namespace util {
  static bool samedim(af::array a, af::array b) {
    return a.dims() == b.dims();
  }
  static bool approx(af::array a, af::array b, float epsilon = 1e-4f) {
    af::array err = af::abs(a - b) > epsilon;
    return af::sum<int>(err) == 0;
  }
  static bool approx(float a,float b, float epsilon = 1e-4f) {
    return abs(a - b) < epsilon;
  }
  static bool nonans(af::array a) {
    return af::sum<int>(af::isNaN(a)) == 0;
  }
  static bool noinfs(af::array a) {
    return af::sum<int>(af::isInf(a)) == 0;
  }
  static bool isnumber(af::array a) {
    return noinfs(a) && nonans(a);
  }
}