#pragma once

#include <arrayfire.h>
#include "serialization.hpp"

namespace dtnn {
  class wb {
  public:
    wb() = default;
    wb(af::array w, af::array b);
    wb(af::dim4 wdim, dim_t bdim);
    wb(af::dim4 wdim, dim_t bdim, float sigma);
    void zero();
    af::array w;
    af::array b;
  };

  template<class Archive> void serialize(Archive & archive, wb & m)
  {
    archive(m.w, m.b);
  }

  void operator+=(wb &lhs, const wb &rhs);
  void operator-=(wb &lhs, const wb &rhs);
  wb operator*(const wb &lhs, const wb &rhs);
  wb operator*(const wb &lhs, float rhs);
  wb operator*(float lhs, const wb &rhs);
  wb operator/(const wb &lhs, const wb &rhs);
  wb operator/(const wb &lhs, float rhs);
  wb operator+(const wb &lhs, const wb &rhs);
  wb operator+(const wb &lhs, float rhs);
  wb operator+(float lhs, const wb &rhs);
  wb operator>(const wb &lhs, float rhs);
  wb operator<(const wb &lhs, float rhs);
  wb operator!(const wb &op);
  wb operator-(const wb &op);
}