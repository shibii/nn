#include "prediction_result.hpp"

namespace dtnn {
  PredictionResult::PredictionResult(af::array output)
    : output_(output)
  {
  }
  std::vector<float> PredictionResult::output_raw() {
    std::vector<float> raw(output_.elements());
    output_.host(raw.data());
    return raw;
  }
  unsigned int PredictionResult::classify() {
    float value;
    unsigned index;
    af::max(&value, &index, output_);
    return index;
  }
  std::vector<uint8_t> PredictionResult::classify(float threshold) {
    auto over_threshold = output_ >= threshold;
    std::vector<uint8_t> raw(output_.elements());
    over_threshold.as(u8).host(raw.data());
    return raw;
  }
  af::array PredictionResult::column_batch(af::array &a) {
    af::dim4 column(a.dims(0) * a.dims(1) * a.dims(2), 1, 1, a.dims(3));
    return af::moddims(a, column);
  }
}