#include "prediction_result.hpp"
#include "arrayfire_util.hpp"

namespace nn {
  PredictionResult::PredictionResult(af::array output)
    : output_(output)
  {
  }
  std::vector<float> PredictionResult::output_raw() const {
    return util::vectorize(output_);
  }
  unsigned int PredictionResult::classify() const {
    float value;
    unsigned index;
    af::max(&value, &index, output_);
    return index;
  }
  std::vector<uint8_t> PredictionResult::classify(float threshold) const {
    auto over_threshold = output_ >= threshold;
    std::vector<uint8_t> raw(output_.elements());
    over_threshold.as(u8).host(raw.data());
    return raw;
  }
}