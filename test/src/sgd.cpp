#include <sstream>
#include "catch.hpp"

#include "techniques/SGD.hpp"

#include "util.hpp"

TEST_CASE("sgd", "[sgd]") {
  float hw[] = { 2, 4, 3, -4 };
  float hb[] = { 1, 2 };
  float hgw[] = { 10, 20, 30, -40 };
  float hgb[] = { -10, 20 };

  auto ow = std::make_shared<nn::OptimizableWeights>();
  ow->weights = {
    af::array(af::dim4(2, 2), hw),
    af::array(af::dim4(2), hb)
  };
  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  auto optimizer = nn::SGD();
  optimizer.attach(ow);
  nn::Hyperparameters hp;
  hp.batch_size = 10;
  hp.learningrate = 0.5f;
  optimizer.optimize(hp);

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
  std::shared_ptr<nn::Optimizer> optimizer;
  optimizer = std::make_shared<nn::SGD>();

  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(optimizer);
  }
  std::string identifier("\"polymorphic_name\": \"nn::SGD\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}