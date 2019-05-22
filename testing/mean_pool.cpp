#include <sstream>
#include "catch.hpp"

#include <cereal/types/vector.hpp>
#include "techniques/mean_pool.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("mean pool", "[mean pool]") {
  float hostinput[] = { 1,2,1,0,0,1,1,1,2,3,1,1,0,0,1,2,0,0,1,
    1,1,1,3,1,1,1,2,0,0,2,0,0,0,0,0,1 };
  float hostexpected[] = { .75,1,.5,1,1,.75,.5,.25,1.5,1.5,1.5,1.75,0,.5,0,.25 };
  af::array expected = af::array(af::dim4(2, 2, 2, 2), hostexpected);
  dtnn::Feed f;
  f.signal = af::array(af::dim4(3, 3, 2, 2), hostinput);
  auto pool = dtnn::MeanPool(2, 2, 1, 1, 0, 0);
  pool.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hosterror[] = { 1,0,0,2,1,1,1,1,0,1,0,0,1,0,1,0 };
  float hostexpectederror[] = { .25,.25,0,.25,.75,.5,0,.5,.5,
  .25,.5,.25,.5,1,.5,.25,.5,.25,
  0,.25,.25,0,.25,.25,0,0,0,
  .25,.25,0,.5,.5,0,.25,.25,0};
  af::array expectederror = af::array(af::dim4(3, 3, 2, 2), hostexpectederror);
  f.signal = af::array(af::dim4(2, 2, 2, 2), hosterror);
  pool.backward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("mean pool serializes", "[mean pool]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto pool = std::make_shared<dtnn::MeanPool>(dtnn::MeanPool(2, 2, 2, 2, 0, 0));
  stages.push_back(pool);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::MeanPool\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}