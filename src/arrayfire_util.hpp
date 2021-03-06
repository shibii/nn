#pragma once

#include <arrayfire.h>
#include <vector>

namespace nn {
namespace util {
static af::array column_batch(const af::array &a) {
  af::dim4 column(a.dims(0) * a.dims(1) * a.dims(2), 1, 1, a.dims(3));
  return af::moddims(a, column);
}
static void randomize_samples(af::array &input, af::array &target) {
  dim_t samples = input.dims(3);
  af::array randoms = af::randu(samples);
  af::array sorted, indices;
  af::sort(sorted, indices, randoms);
  input = input(af::span, af::span, af::span, indices);
  target = target(af::span, af::span, af::span, indices);
}
static af::array unzero(const af::array &a, float r = 1e-6f) {
  return (af::abs(a) >= r) * a + (af::abs(a) < r) * r;
}
static std::vector<float> vectorize(const af::array &a) {
  std::vector<float> vec(a.elements());
  a.host(vec.data());
  return vec;
}
static af::array add(const af::array &lhs, const af::array &rhs) {
  return lhs + rhs;
}
static af::array sub(const af::array &lhs, const af::array &rhs) {
  return lhs - rhs;
}
static af::array div(const af::array &lhs, const af::array &rhs) {
  return lhs / rhs;
}
static af::array mul(const af::array &lhs, const af::array &rhs) {
  return lhs * rhs;
}
}  // namespace util
}  // namespace nn