#include <sstream>
#include "catch.hpp"

#include "cereal_archives.hpp"
#include "optimizable_weights.hpp"

#include "util.hpp"

TEST_CASE("optimizable weights serializes", "[optimizable weights]") {
  nn::wb ow = { af::randu(af::dim4(2)), af::randu(af::dim4(1)) };
  nn::wb og = { af::randu(af::dim4(2)), af::randu(af::dim4(1)) };
  nn::OptimizableWeights oweights = { ow, og };
  std::stringstream stream;
  {
    cereal::JSONOutputArchive oarchive(stream);
    oarchive(oweights);
  }
  nn::OptimizableWeights iweights;
  {
    cereal::JSONInputArchive iarchive(stream);
    iarchive(iweights);
  }
  REQUIRE(util::approx(iweights.weights.w, oweights.weights.w));
  REQUIRE(util::approx(iweights.weights.b, oweights.weights.b));
  REQUIRE(util::approx(iweights.gradient.w, oweights.gradient.w));
  REQUIRE(util::approx(iweights.gradient.b, oweights.gradient.b));
}