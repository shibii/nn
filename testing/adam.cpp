#include <sstream>
#include "catch.hpp"

#include "techniques/adam.hpp"

#include "util.hpp"

TEST_CASE("adam", "[adam]") {
  float hw[] = { 2, 4, 3, -4 };
  float hb[] = { 1, 2 };
  float hgw[] = { 10, 20, -10, -20 };
  float hgb[] = { 10, 30 };

  auto ow = std::make_shared<dtnn::OptimizableWeights>();
  ow->weights = {
    af::array(af::dim4(2, 2), hw),
    af::array(af::dim4(2), hb)
  };
  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  auto optimizer = dtnn::Adam(0.8f, 0.9f);
  optimizer.attach(ow);
  dtnn::Hyperparameters hp;
  hp.batch_size = 10;
  hp.learningrate = 0.5f;
  optimizer.optimize(hp);

  float h_w1[] = { 1.5, 3.5, 3.5, -3.5 };
  float h_b1[] = { 0.5, 1.5 };
  af::array w1 = af::array(af::dim4(2, 2), h_w1);
  af::array b1 = af::array(af::dim4(2), h_b1);

  REQUIRE(util::approx(ow->weights.w, w1));
  REQUIRE(util::approx(ow->weights.b, b1));

  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  float h_w2[] = { 1, 3, 4, -3 };
  float h_b2[] = { 0, 1 };
  af::array w2 = af::array(af::dim4(2, 2), h_w2);
  af::array b2 = af::array(af::dim4(2), h_b2);

  optimizer.optimize(hp);

  REQUIRE(util::approx(ow->weights.w, w2));
  REQUIRE(util::approx(ow->weights.b, b2));
}

TEST_CASE("adam serializes", "[adam]") {
  std::shared_ptr<dtnn::Optimizer> optimizer;
  optimizer = std::make_shared<dtnn::Adam>();

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
  std::string identifier("\"polymorphic_name\": \"dtnn::Adam\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}