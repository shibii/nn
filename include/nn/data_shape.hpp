#pragma once

#include <arrayfire.h>

namespace nn {
class DataShape {
 public:
  DataShape(long long dim0, long long dim1, long long dim2, long long dim3)
      : dim0_(dim0), dim1_(dim1), dim2_(dim2), dim3_(dim3) {}
  operator af::dim4() const { return af::dim4(dim0_, dim1_, dim2_, dim3_); }

 private:
  long long dim0_;
  long long dim1_;
  long long dim2_;
  long long dim3_;
};
}  // namespace nn