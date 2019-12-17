#include <sstream>
#include "catch.hpp"

#include <cereal/types/vector.hpp>
#include "techniques/logistic.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("logistic", "[logistic]") {
  float hostinput[] = { -1e+20f, -1.f, 0.f, 1.f, 1e+20f };
  float hostexpected[] = { 0.f, .2689f, .5f, .7310f, 1.f};
  af::array expected = af::array(af::dim4(5), hostexpected);
  nn::Feed f;
  f.signal = af::array(af::dim4(5), hostinput);
  auto logistic = nn::Logistic();
  logistic.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hosterror[] = { 3.f, 6.f, -5.f, 0.f, -.5f };
  float hostexpectederror[] = { 0.f, 1.1796f, -1.25f, 0.f, 0.f };
  af::array expectederror = af::array(af::dim4(5), hostexpectederror);
  f.signal = af::array(af::dim4(5), hosterror);
  logistic.backward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("logistic serializes", "[logistic]") {
  std::vector<std::shared_ptr<nn::PropagationStage>> stages;
  auto af = std::make_shared<nn::Logistic>(nn::Logistic());
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"nn::Logistic\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}