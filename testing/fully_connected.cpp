#include "catch.hpp"
#include <dtnn.hpp>
#include "util.hpp"
#include <sstream>

TEST_CASE("fully connected", "[fully connected]") {
  float hweights[] = { 1, -1, 2, 1 };
  float hbias[] = { 1, 2 };

  dtnn::Feed f;
  float hinput[] = { 0, 1, -1, -2 };
  f.signal = af::array(af::dim4(2, 1, 1, 2), hinput);

  auto fc = dtnn::FullyConnected(2);
  auto param = fc.init(f);
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
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  auto af = std::make_shared<dtnn::FullyConnected>(dtnn::FullyConnected());
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::FullyConnected\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}