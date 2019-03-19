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
}