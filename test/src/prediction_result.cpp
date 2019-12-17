#include <vector>
#include "catch.hpp"

#include "prediction_result.hpp"

#include "util.hpp"

TEST_CASE("prediction result", "[prediction result]") {
  std::vector<float> h_ouput{ 1, 2, 3, 4, 2, 2, 2, 2, 4, 4, 5, 4 };
  af::array output = af::array(af::dim4(2, 2, 1, 3), h_ouput.data());
  nn::PredictionResult result(output);

  REQUIRE(h_ouput == result.output_raw());
  REQUIRE(10 == result.classify());

  std::vector<uint8_t> classification{ 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1 };
  REQUIRE(classification == result.classify(4.f));

}