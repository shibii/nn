#pragma once

#include "sample_provider.hpp"

namespace dtnn {
  class BlobProvider : public SampleProvider {
  public:
    BlobProvider(float* inputs, float* targets, af::dim4 inputdim, af::dim4 targetdim);
    Samples batch(dim_t size) override;
  private:
    af::array inputs_;
    af::array targets_;
    dim_t location_;
  };
}