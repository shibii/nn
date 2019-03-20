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
}