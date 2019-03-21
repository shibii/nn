#include <sstream>
#include "catch.hpp"

#include <cereal/types/vector.hpp>
#include "techniques/max_pool.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("max pool", "[max pool]") {
  float hostinput[] = { 1,2,1,0,0,1,1,1,2,3,1,1,0,0,1,2,0,0,1,
    1,1,1,3,1,1,1,2,0,0,2,0,0,0,0,0,1 };
  float hostexpected[] = { 2,2,1,2,3,1,2,1,3,3,3,3,0,2,0,1 };
  af::array expected = af::array(af::dim4(2,2,2,2), hostexpected);
  dtnn::Feed f;
  f.signal = af::array(af::dim4(3,3,2,2), hostinput);
  auto pool = dtnn::MaxPool(2,2,1,1,0,0);
  pool.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hosterror[] = { 1,0,0,2,1,1,1,1,0,1,0,0,1,0,1,0 };
  float hostexpectederror[] = { 0,1,0,0,0,0,0,0,2,1,1,0,0,0,1,1,0,0,
    0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0 };
  af::array expectederror = af::array(af::dim4(3,3,2,2), hostexpectederror);
  f.signal = af::array(af::dim4(2,2,2,2), hosterror);
  pool.backward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("max pool serializes", "[max pool]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto pool = std::make_shared<dtnn::MaxPool>(dtnn::MaxPool(2,2,2,2,0,0));
  stages.push_back(pool);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::MaxPool\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}