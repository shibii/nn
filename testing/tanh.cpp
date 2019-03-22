#include <sstream>
#include "catch.hpp"

#include <cereal/types/vector.hpp>
#include "techniques/tanh.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("tanh", "[tanh]") {
  float hostinput[] = { -1e+20f, -1.f, 0.f, 1.f, 1e+20f };
  float hostexpected[] = { -1.f, -.7616f, 0.f, .7616f, 1.f };
  af::array expected = af::array(af::dim4(5), hostexpected);
  dtnn::Feed f;
  f.signal = af::array(af::dim4(5), hostinput);
  auto tanh = dtnn::Tanh();
  tanh.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hosterror[] = { 3.f, 6.f, -5.f, 0.f, -.5f };
  float hostexpectederror[] = { 0.f, 2.5198f, -5.f, 0.f, 0.f };
  af::array expectederror = af::array(af::dim4(5), hostexpectederror);
  f.signal = af::array(af::dim4(5), hosterror);
  tanh.backward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("tanh serializes", "[tanh]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto af = std::make_shared<dtnn::Tanh>(dtnn::Tanh());
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::Tanh\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}