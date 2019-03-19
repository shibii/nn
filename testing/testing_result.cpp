#include <cstdlib>
#include "catch.hpp"

#include "testing_result.hpp"

#include "util.hpp"

TEST_CASE("testing result", "[testing result]") {
  float h_ouput[] = { 1, 2, 3, 4, 2, 2, 2, 2, 4, 4, 4, 4 };
  float h_target[] = { 2, 2, 2, 2, 4, 4, 4, 4, 1, 2, 3, 4 };
  float h_loss[] = { 0.5, 0, 0.5, 2, 2, 2, 2, 2, 4.5, 2, 0.5, 0 };
  af::array output = af::array(af::dim4(2, 2, 1, 3), h_ouput);
  af::array target = af::array(af::dim4(2, 2, 1, 3), h_target);
  af::array loss = af::array(af::dim4(2, 2, 1, 3), h_loss);
  dtnn::TestingResult result(output, target, loss);

  float epsilon = 1e-4;

  REQUIRE(result.loss() == 6.f);
  REQUIRE(abs(result.rmse() - 1.6985f) < epsilon);

  //REQUIRE(abs(result.loss() - 6.f) < epsilon);
}