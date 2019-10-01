#pragma once

#include <arrayfire.h>

#include "arrayfire_serialization.hpp"

namespace nn {
  class wb {
  public:
    wb() = default;
    wb(af::array w, af::array b);
    wb(af::dim4 wdim, af::dim4 bdim);
    wb(af::dim4 wdim, af::dim4 bdim, float sigma);
    void zero();
    wb pow(int p) const;
    wb sqrt() const;
    template <class Archive> void serialize(Archive &ar) { ar(w, b); }

    af::array w;
    af::array b;
  };

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