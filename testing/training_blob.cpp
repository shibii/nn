#include <cstdlib>
#include <vector>
#include "catch.hpp"

#include "training_blob.hpp"
#include "util.hpp"

TEST_CASE("training blob", "[training blob]") {
  std::vector<float> inputs = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> targets = { 0.f, 1.f, 2.f, 3.f };
  auto provider = dtnn::TrainingBlob(inputs.data(), targets.data(), af::dim4(2, 1, 1, 4), af::dim4(1, 1, 1, 4));

  std::vector<float> h_input1{ 0.f, 1.f, 2.f, 3.f };
  af::array input1 = af::array(af::dim4(2, 1, 1, 2), h_input1.data());

  std::vector<float> h_target1{ 0.f, 1.f };
  af::array target1 = af::array(af::dim4(1, 1, 1, 2), h_target1.data());

  auto batch = provider.batch(2);
  REQUIRE(util::approx(batch.inputs, input1));
  REQUIRE(util::approx(batch.targets, target1));

  std::vector<float> h_input2{ 4.f, 5.f, 6.f, 7.f };
  af::array input2 = af::array(af::dim4(2, 1, 1, 2), h_input2.data());

  std::vector<float> h_target2{ 2.f, 3.f };
  af::array target2 = af::array(af::dim4(1, 1, 1, 2), h_target2.data());

  batch = provider.batch(2);
  REQUIRE(util::approx(batch.inputs, input2));
  REQUIRE(util::approx(batch.targets, target2));

  batch = provider.batch(2);
  REQUIRE(util::approx(batch.inputs, input1));
  REQUIRE(util::approx(batch.targets, target1));
}