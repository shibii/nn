#include <sstream>
#include "catch.hpp"

#include "cereal_archives.hpp"
#include "wb.hpp"

#include "util.hpp"

TEST_CASE("wb serializes", "[wb]") {
  dtnn::wb owb = { af::randu(af::dim4(2)), af::randu(af::dim4(1)) };
  std::stringstream stream;
  {
    cereal::JSONOutputArchive oarchive(stream);
    oarchive(owb);
  }
  dtnn::wb iwb;
  {
    cereal::JSONInputArchive iarchive(stream);
    iarchive(iwb);
  }
  REQUIRE(util::approx(iwb.w, owb.w));
  REQUIRE(util::approx(iwb.b, owb.b));
}