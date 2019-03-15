#include "catch.hpp"
#include <dtnn.hpp>
#include "util.hpp"
#include <sstream>

TEST_CASE("sgd", "[sgd]") {
  float hw[] = { 2, 4, 3, -4 };
  float hb[] = { 1, 2 };
  float hgw[] = { 1, 2, 3, -4 };
  float hgb[] = { -1, 2 };

  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  weights->w = af::array(af::dim4(2, 2), hw);
  weights->b = af::array(af::dim4(2), hb);
  auto gradient = std::make_shared<dtnn::wb>(dtnn::wb());
  gradient->w = af::array(af::dim4(2, 2), hgw);
  gradient->b = af::array(af::dim4(2), hgb);
  auto optimizer = dtnn::SGD(0.5f);
  optimizer.attach(weights, gradient);
  optimizer.optimize();

  float hexpectedw[] = { 1.5, 3, 1.5, -2 };
  float hexpectedb[] = { 1.5, 1 };
  af::array expectedw = af::array(af::dim4(2, 2), hexpectedw);
  af::array expectedb = af::array(af::dim4(2), hexpectedb);

  REQUIRE(util::isnumber(weights->w));
  REQUIRE(util::isnumber(weights->b));
  REQUIRE(util::samedim(weights->w, expectedw));
  REQUIRE(util::samedim(weights->b, expectedb));
  REQUIRE(util::approx(weights->w, expectedw));
  REQUIRE(util::approx(weights->b, expectedb));
}

//TEST_CASE("sgd serializes", "[sgd]") {
//  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
//  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
//  auto af = std::make_shared<dtnn::FullyConnected>(dtnn::FullyConnected(weights));
//  stages.push_back(af);
//  std::ostringstream ostream;
//  {
//    cereal::JSONOutputArchive oarchive(ostream);
//    oarchive(stages);
//  }
//  std::string identifier("\"polymorphic_name\": \"dtnn::FullyConnected\"");
//  REQUIRE(ostream.str().find(identifier) != std::string::npos);
//}