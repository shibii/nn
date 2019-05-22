#include <sstream>
#include "catch.hpp"

#include "techniques/momentum.hpp"

#include "util.hpp"

TEST_CASE("momentum", "[momentum]") {
  float hw[] = { 2, 4, 3, -4 };
  float hb[] = { 1, 2 };
  float hgw[] = { 10, 20, 30, -40 };
  float hgb[] = { -10, 20 };

  auto ow = std::make_shared<dtnn::OptimizableWeights>();
  ow->weights = {
    af::array(af::dim4(2, 2), hw),
    af::array(af::dim4(2), hb)
  };
  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  auto optimizer = dtnn::Momentum(0.1f);
  optimizer.attach(ow);
  dtnn::Hyperparameters hp;
  hp.batch_size = 10;
  hp.learningrate = 0.5f;
  optimizer.optimize(hp);

  float h_w1[] = { 1.5, 3, 1.5, -2 };
  float h_b1[] = { 1.5, 1 };
  af::array w1 = af::array(af::dim4(2, 2), h_w1);
  af::array b1 = af::array(af::dim4(2), h_b1);

  REQUIRE(util::approx(ow->weights.w, w1));
  REQUIRE(util::approx(ow->weights.b, b1));

  float h_w2[] = { 0.55, 1.1, -1.35, 1.8 };
  float h_b2[] = { 2.45, -0.9 };
  af::array w2 = af::array(af::dim4(2, 2), h_w2);
  af::array b2 = af::array(af::dim4(2), h_b2);

  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };
  optimizer.optimize(hp);

  REQUIRE(util::approx(ow->weights.w, w2));
  REQUIRE(util::approx(ow->weights.b, b2));
}

TEST_CASE("momentum serializes", "[momentum]") {
  std::shared_ptr<dtnn::Optimizer> optimizer;
  optimizer = std::make_shared<dtnn::Momentum>();

  auto ow = std::make_shared<dtnn::OptimizableWeights>();
  ow->weights = {
    af::randu(af::dim4(2, 2)),
    af::randu(af::dim4(2))
  };
  ow->gradient = {
    af::randu(af::dim4(2, 2)),
    af::randu(af::dim4(2))
  };
  optimizer->attach(ow);

  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(optimizer);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::Momentum\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}