#include "wb.hpp"
#include "arrayfire_util.hpp"

namespace dtnn {
  wb::wb(af::array w, af::array b) : w(w), b(b) {
  }
  wb::wb(af::dim4 wdim, af::dim4 bdim) {
    w = af::constant(0.f, wdim);
    b = af::constant(0.f, bdim);
  }
  wb::wb(af::dim4 wdim, af::dim4 bdim, float sigma) {
    w = af::randn(wdim) * sigma;
    b = af::constant(0.f, bdim);
  }
  void wb::zero() {
    w = af::constant(0.f, w.dims());
    b = af::constant(0.f, b.dims());
  }
  wb wb::pow(int p) const {
    return { af::pow(w, p) , af::pow(b, p) };
  }
  wb wb::sqrt() const {
    return { af::sqrt(util::unzero(w)), af::sqrt(util::unzero(b)) };
  }
  void operator+=(wb& lhs, const wb& rhs) {
    lhs.w = lhs.w + rhs.w;
    lhs.b = lhs.b + rhs.b;
  }
  void operator-=(wb& lhs, const wb& rhs) {
    lhs.w = lhs.w - rhs.w;
    lhs.b = lhs.b - rhs.b;
  }
  wb operator*(const wb& lhs, const wb& rhs) {
    return { lhs.w * rhs.w, lhs.b * rhs.b };
  }
  wb operator*(const wb& lhs, float rhs) {
    return { lhs.w * rhs, lhs.b * rhs };
  }
  wb operator*(float lhs, const wb& rhs) {
    return { rhs.w * lhs, rhs.b * lhs };
  }
  wb operator/(const wb& lhs, const wb& rhs) {
    return { lhs.w / rhs.w, lhs.b / rhs.b };
  }
  wb operator/(const wb& lhs, float rhs) {
    return { lhs.w / rhs, lhs.b / rhs };
  }
  wb operator+(const wb& lhs, const wb& rhs) {
    return { lhs.w + rhs.w, lhs.b + rhs.b };
  }
  wb operator+(const wb& lhs, float rhs) {
    return { lhs.w + rhs, lhs.b + rhs };
  }
  wb operator+(float lhs, const wb& rhs) {
    return { rhs.w + lhs, rhs.b + lhs };
  }
  wb operator>(const wb& lhs, float rhs) {
    return { lhs.w > rhs, lhs.b > rhs };
  }
  wb operator<(const wb& lhs, float rhs) {
    return { lhs.w < rhs, lhs.b < rhs };
  }
  wb operator!(const wb& op) { return { !op.w, !op.b }; }
  wb operator-(const wb& op) { return { -op.w, -op.b }; }
}