#include <cstdlib>
#include <vector>
#include "catch.hpp"

#include "testing_result.hpp"
#include "util.hpp"

TEST_CASE("testing result", "[testing result]") {
  std::vector<float> h_ouput{ 1, 2, 3, 4, 2, 2, 2, 2, 4, 4, 4, 4 };
  std::vector<float> h_target{ 2, 2, 2, 2, 4, 4, 4, 4, 1, 2, 3, 4 };
  std::vector<float> h_loss{ 0.5, 0, 0.5, 2, 2, 2, 2, 2, 4.5, 2, 0.5, 0 };
  af::array output = af::array(af::dim4(2, 2, 1, 3), h_ouput.data());
  af::array target = af::array(af::dim4(2, 2, 1, 3), h_target.data());
  af::array loss = af::array(af::dim4(2, 2, 1, 3), h_loss.data());
  dtnn::TestingResult result(output, target, loss);

  REQUIRE(h_ouput == result.output_raw());
  REQUIRE(h_target == result.target_raw());
  REQUIRE(h_loss == result.loss_raw());

  REQUIRE(util::approx(result.loss(), 6.f));
  REQUIRE(util::approx(result.rmse(), 1.6985f));

  h_ouput = { .3f, .1f, .1f, .2f,
    .7f, .6f, .1f, .1f,
    .6f, .7f, .6f, .1f,
    .1f, .2f, .1f, .0f,
    .0f, .7f, .8f, .1f,
    .1f, .2f, .3f, .0f};
  h_target = { 1.f,0.f,0.f,0.f,
    0.f,1.f,0.f,0.f,
    0.f,1.f,0.f,0.f,
    1.f,0.f,0.f,0.f,
    0.f,0.f,0.f,1.f,
    1.f,0.f,0.f,0.f };
  output = af::array(af::dim4(2, 2, 1, 6), h_ouput.data());
  target = af::array(af::dim4(2, 2, 1, 6), h_target.data());
  result = { output, target, loss };

  REQUIRE(util::approx(0.3333f, result.accuracy()));


  h_ouput = { .3f, .1f, .1f, .2f };
  h_target = { 1.f,0.f,0.f,0.f };
  output = af::array(af::dim4(2, 2, 1, 1), h_ouput.data());
  target = af::array(af::dim4(2, 2, 1, 1), h_target.data());
  result = { output, target, loss };
  REQUIRE(util::approx(1.f, result.accuracy()));


  h_ouput = { .7f, .6f, .1f, .1f };
  h_target = { 0.f,1.f,0.f,0.f };
  output = af::array(af::dim4(2, 2, 1, 1), h_ouput.data());
  target = af::array(af::dim4(2, 2, 1, 1), h_target.data());
  result = { output, target, loss };
  REQUIRE(util::approx(0.f, result.accuracy()));

  h_ouput = { .7, .2, .6, .1, .3, .7, .8, .2, .7, .1, .7, .6 };
  h_target = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0 };
  output = af::array(af::dim4(2, 2, 1, 3), h_ouput.data());
  target = af::array(af::dim4(2, 2, 1, 3), h_target.data());
  result = { output, target, loss };

  REQUIRE(util::approx(.7143f, result.precision(0.6f)));
  REQUIRE(util::approx(.8333f, result.recall(0.6f)));
  REQUIRE(util::approx(.7692f, result.f1(0.6f)));
  REQUIRE(util::approx(.75f, result.accuracy(0.6f)));
  REQUIRE(util::approx(.6667f, result.specificity(0.6f)));
}