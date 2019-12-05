#include "catch.hpp"
#include <vector>
#include "util.hpp"
#include "arrayfire_util.hpp"

TEST_CASE("arrayfire_util", "[arrayfire_util]") {
  float ha[] = {-2.f, 0.5f, -0.1f, 0.1f};
  auto a = af::array(af::dim4(4), ha);
  float hc[] = {-2.f, 0.5f, 0.2f, 0.2f};
  auto c = af::array(af::dim4(4), hc);
  REQUIRE(util::approx(c, nn::util::unzero(a, 0.2f)));
}