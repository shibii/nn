#include "testing_result.hpp"
#include "arrayfire_util.hpp"

namespace nn {
TestingResult::TestingResult(af::array output, af::array target, af::array loss)
    : output_(output), target_(target), loss_(loss) {}
std::vector<float> TestingResult::output_raw() const {
  return util::vectorize(output_);
}
std::vector<float> TestingResult::target_raw() const {
  return util::vectorize(target_);
}
std::vector<float> TestingResult::loss_raw() const {
  return util::vectorize(loss_);
}
float TestingResult::loss() const {
  af::array metric = util::column_batch(loss_);
  metric = af::sum(metric, 0);
  return af::mean<float>(metric);
}
float TestingResult::rmse() const {
  af::array metric = output_ - target_;
  metric = util::column_batch(metric);
  metric = af::pow(metric, 2);
  metric = af::mean(metric, 0);
  metric = af::sqrt(metric);
  return af::mean<float>(metric);
}
float TestingResult::precision(float threshold) const {
  float tp = true_positive(threshold);
  float fp = false_positive(threshold);
  return tp / (tp + fp);
}
float TestingResult::recall(float threshold) const {
  float tp = true_positive(threshold);
  float fn = false_negative(threshold);
  return tp / (tp + fn);
}
float TestingResult::f1(float threshold) const {
  float p = precision(threshold);
  float r = recall(threshold);
  return (2 * p * r) / (p + r);
}
float TestingResult::specificity(float threshold) const {
  float tn = true_negative(threshold);
  float fp = false_positive(threshold);
  return tn / (tn + fp);
}
float TestingResult::accuracy(float threshold) const {
  af::array over_threshold = output_ >= threshold;
  af::array correct = over_threshold == target_;
  return af::count<float>(correct) / correct.elements();
}
float TestingResult::true_positive(float threshold) const {
  af::array positive = output_ >= threshold;
  return af::count<float>(positive && target_);
}
float TestingResult::true_negative(float threshold) const {
  af::array positive = output_ >= threshold;
  return af::count<float>(!positive && !target_);
}
float TestingResult::false_positive(float threshold) const {
  af::array positive = output_ >= threshold;
  return af::count<float>(positive && !target_);
}
float TestingResult::false_negative(float threshold) const {
  af::array positive = output_ >= threshold;
  return af::count<float>(!positive && target_);
}
float TestingResult::accuracy() const {
  af::array output_column = util::column_batch(output_);
  af::array target_column = util::column_batch(target_);
  af::array values, outputIdx, targetIdx;
  af::max(values, outputIdx, output_column, 0);
  af::max(values, targetIdx, target_column, 0);
  return af::mean<float>(outputIdx == targetIdx);
}
}  // namespace nn