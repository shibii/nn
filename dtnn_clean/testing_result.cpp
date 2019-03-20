#include "testing_result.hpp"

namespace dtnn {
  TestingResult::TestingResult(af::array output, af::array target, af::array loss)
    : output_(output), target_(target), loss_(loss)
  {}
  std::vector<float> TestingResult::output_raw() {
    std::vector<float> raw(output_.elements());
    output_.host(raw.data());
    return raw;
  }
  std::vector<float> TestingResult::target_raw() {
    std::vector<float> raw(target_.elements());
    target_.host(raw.data());
    return raw;
  }
  std::vector<float> TestingResult::loss_raw() {
    std::vector<float> raw(loss_.elements());
    loss_.host(raw.data());
    return raw;
  }
  float TestingResult::loss() {
    af::array metric = column_batch(loss_);
    metric = af::sum(metric, 0);
    return af::mean<float>(metric);
  }
  float TestingResult::rmse() {
    af::array metric = output_ - target_;
    metric = column_batch(metric);
    metric = af::pow(metric, 2);
    metric = af::mean(metric, 0);
    metric = af::sqrt(metric);
    return af::mean<float>(metric);
  }
  float TestingResult::accuracy() {
    af::array output_column = column_batch(output_);
    af::array target_column = column_batch(target_);
    af::array output_max, output_maxIdx;
    af::max(output_max, output_maxIdx, output_column, 0);
    af::array target_max, target_maxIdx;
    af::max(target_max, target_maxIdx, target_column, 0);
    return af::sum<float>(output_maxIdx == target_maxIdx) / output_.dims(3);
  }
  float TestingResult::accuracy(float threshold) {
    af::array over_threshold = output_ >= threshold;
    af::array correct = over_threshold == target_;
    return af::sum<float>(correct) / correct.elements();
  }
  af::array TestingResult::column_batch(af::array &a) {
    af::dim4 column(a.dims(0) * a.dims(1) * a.dims(2), 1, 1, a.dims(3));
    return af::moddims(a, column);
  }
}