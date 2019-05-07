#include <sstream>
#include "catch.hpp"

#include "techniques/SGD.hpp"

#include "util.hpp"

TEST_CASE("sgd", "[sgd]") {
  float hw[] = { 2, 4, 3, -4 };
  float hb[] = { 1, 2 };
  float hgw[] = { 1, 2, 3, -4 };
  float hgb[] = { -1, 2 };

  auto ow = std::make_shared<dtnn::OptimizableWeights>();
  ow->weights = {
    af::array(af::dim4(2, 2), hw),
    af::array(af::dim4(2), hb)
  };
  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  auto optimizer = dtnn::SGD(0.5f);
  optimizer.attach(ow);
  optimizer.optimize(1);

  float hexpectedw[] = { 1.5, 3, 1.5, -2 };
  float hexpectedb[] = { 1.5, 1 };
  af::array expectedw = af::array(af::dim4(2, 2), hexpectedw);
  af::array expectedb = af::array(af::dim4(2), hexpectedb);

  REQUIRE(util::isnumber(ow->weights.w));
  REQUIRE(util::isnumber(ow->weights.b));
  REQUIRE(util::samedim(ow->weights.w, expectedw));
  REQUIRE(util::samedim(ow->weights.b, expectedb));
  REQUIRE(util::approx(ow->weights.w, expectedw));
  REQUIRE(util::approx(ow->weights.b, expectedb));
}

TEST_CASE("sgd serializes", "[sgd]") {
  std::shared_ptr<dtnn::Optimizer> optimizer;
  optimizer = std::make_shared<dtnn::SGD>(1.f);

  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(optimizer);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::SGD\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}