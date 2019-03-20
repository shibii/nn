#include "testing_result.hpp"
#include "arrayfire_util.hpp"

namespace dtnn {
  TestingResult::TestingResult(af::array output, af::array target, af::array loss)
    : output_(output), target_(target), loss_(loss)
  {}
  std::vector<float> TestingResult::output_raw() {
    return util::vectorize(output_);
  }
  std::vector<float> TestingResult::target_raw() {
    return util::vectorize(target_);
  }
  std::vector<float> TestingResult::loss_raw() {
    return util::vectorize(loss_);
  }
  float TestingResult::loss() {
    af::array metric = util::column_batch(loss_);
    metric = af::sum(metric, 0);
    return af::mean<float>(metric);
  }
  float TestingResult::rmse() {
    af::array metric = output_ - target_;
    metric = util::column_batch(metric);
    metric = af::pow(metric, 2);
    metric = af::mean(metric, 0);
    metric = af::sqrt(metric);
    return af::mean<float>(metric);
  }
  float TestingResult::precision(float threshold) {
    af::array predicted_positive = output_ >= threshold;
    af::array true_positive = predicted_positive * target_;
    return af::sum<float>(true_positive) / af::sum<float>(predicted_positive);
  }
  float TestingResult::accuracy() {
    af::array output_column = util::column_batch(output_);
    af::array target_column = util::column_batch(target_);
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
}