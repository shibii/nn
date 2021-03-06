#include <sstream>
#include "catch.hpp"

#include "techniques/fully_connected.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("fully connected", "[fully connected]") {
  float hweights[] = { 1, -1, 2, 1 };
  float hbias[] = { 1, 2 };

  nn::Feed f;
  float hinput[] = { 0, 1, -1, -2 };
  f.signal = af::array(af::dim4(2, 1, 1, 2), hinput);

  auto fc = nn::FullyConnected(2);
  auto param = fc.init(f.signal.dims());
  param->weights = {
    af::array(af::dim4(2, 2), hweights),
    af::array(af::dim4(2), hbias)
  };
  param->gradient = {
    af::constant(0.f, af::dim4(2, 2)),
    af::constant(0.f, af::dim4(2))
  };

  fc.forward(f);

  float hexpected[] = { 3, 3, -4, 1 };
  af::array expected = af::array(af::dim4(2, 1, 1, 2), hexpected);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hexpectedgw[] = { -2, -2, -3, -5 };
  af::array expectedgw = af::array(af::dim4(2, 2), hexpectedgw);
  float hexpectedgb[] = { 3, 1 };
  af::array expectedgb = af::array(af::dim4(2), hexpectedgb);

  float herror[] = { 1, -1, 2, 2 };
  f.signal = af::array(af::dim4(2, 1, 1, 2), herror);
  fc.backward(f);
  REQUIRE(util::isnumber(param->gradient.w));
  REQUIRE(util::isnumber(param->gradient.b));
  REQUIRE(util::approx(param->gradient.w, expectedgw));
  REQUIRE(util::approx(param->gradient.b, expectedgb));

  float hexpectederror[] = { 2, 1, 0, 6 };
  af::array expectederror = af::array(af::dim4(2, 1, 1, 2), hexpectederror);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("fully connected serializes", "[fully connected]") {
  std::vector<std::shared_ptr<nn::PropagationStage>> stages;
  auto weights = std::make_shared<nn::wb>(nn::wb());
  auto fc = std::make_shared<nn::FullyConnected>(nn::FullyConnected(2));

  nn::Feed f;
  f.signal = af::constant(0.f, af::dim4(2, 1, 1, 2));
  fc->init(f.signal.dims());

  stages.push_back(fc);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"nn::FullyConnected\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}