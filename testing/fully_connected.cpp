#include "catch.hpp"
#include <dtnn.hpp>
#include "util.hpp"
#include <sstream>

TEST_CASE("fullyconnected forward", "[fullyconnected]") {
  float hweights[] = { 1, -1, 2, 1 };
  float hbias[] = { 1, 2 };
  float hinput[] = { 0, 1, -1, -2 };

  dtnn::Feed f;
  f.signal = af::array(af::dim4(2, 1, 1, 2), hinput);
  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  weights->w = af::array(af::dim4(2, 2), hweights);
  weights->b = af::array(af::dim4(2), hbias);
  auto gradient = std::make_shared<dtnn::wb>(dtnn::wb());
  gradient->w = af::constant(0.f, af::dim4(2, 2));
  gradient->b = af::constant(0.f, af::dim4(2));
  auto fc = dtnn::FullyConnected(weights, gradient);
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

  REQUIRE(util::isnumber(fc.gradient_->w));
  REQUIRE(util::isnumber(fc.gradient_->b));
  REQUIRE(util::approx(fc.gradient_->w, expectedgw));
  REQUIRE(util::approx(fc.gradient_->b, expectedgb));

  float hexpectederror[] = { 2, 1, 0, 6 };
  af::array expectederror = af::array(af::dim4(2, 1, 1, 2), hexpectederror);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("fullyconnected serializes", "[fullyconnected]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  auto af = std::make_shared<dtnn::FullyConnected>(dtnn::FullyConnected(weights));
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::FullyConnected\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}