#include <sstream>
#include "catch.hpp"

#include <cereal/types/vector.hpp>
#include "techniques/lrel.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("lrel", "[lrel]") {
  float hostinput[] = { -1e+20f, -1.f, 0.f, 1.f, 1e+20f };
  float hostexpected[] = { -1e+19f, -.1f, 0.f, 1.f, 1e+20f };
  af::array expected = af::array(af::dim4(5), hostexpected);
  nn::Feed f;
  f.signal = af::array(af::dim4(5), hostinput);
  auto lrel = nn::LReL(0.1f);
  lrel.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hosterror[] = { 3.f, 6.f, -5.f, 1.f, -.5f };
  float hostexpectederror[] = { .3f, .6f, -.5f, 1.f, -.5f };
  af::array expectederror = af::array(af::dim4(5), hostexpectederror);
  f.signal = af::array(af::dim4(5), hosterror);
  lrel.backward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("lrel serializes", "[lrel]") {
  std::vector<std::shared_ptr<nn::PropagationStage>> stages;
  auto af = std::make_shared<nn::LReL>(nn::LReL());
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"nn::LReL\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}