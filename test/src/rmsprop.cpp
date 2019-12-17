#include <sstream>
#include "catch.hpp"

#include "techniques/rmsprop.hpp"

#include "util.hpp"

TEST_CASE("rmsprop", "[rmsprop]") {
  float hw[] = { 2, 4, 3, -4 };
  float hb[] = { 1, 2 };
  float hgw[] = { 10, 20, -10, -20 };
  float hgb[] = { 10, 30 };

  auto ow = std::make_shared<nn::OptimizableWeights>();
  ow->weights = {
    af::array(af::dim4(2, 2), hw),
    af::array(af::dim4(2), hb)
  };
  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  auto optimizer = nn::RMSprop(0.2f);
  optimizer.attach(ow);
  nn::Hyperparameters hp;
  hp.batch_size = 10;
  hp.learningrate = 0.5f;
  optimizer.optimize(hp);

  float h_w1[] = { 0.882f, 2.882f, 4.118f, -2.882f };
  float h_b1[] = { -0.118f, 0.882f };
  af::array w1 = af::array(af::dim4(2, 2), h_w1);
  af::array b1 = af::array(af::dim4(2), h_b1);

  REQUIRE(util::approx(ow->weights.w, w1));
  REQUIRE(util::approx(ow->weights.b, b1));

  ow->gradient = {
    af::array(af::dim4(2, 2), hgw),
    af::array(af::dim4(2), hgb)
  };

  float h_w2[] = { 0.0486f, 2.0486f, 4.9513f, -2.0486f };
  float h_b2[] = { -0.9514f, 0.0486f };
  af::array w2 = af::array(af::dim4(2, 2), h_w2);
  af::array b2 = af::array(af::dim4(2), h_b2);

  optimizer.optimize(hp);

  REQUIRE(util::approx(ow->weights.w, w2));
  REQUIRE(util::approx(ow->weights.b, b2));
}

TEST_CASE("rmsprop serializes", "[rmsprop]") {
  std::shared_ptr<nn::Optimizer> optimizer;
  optimizer = std::make_shared<nn::RMSprop>();

  auto ow = std::make_shared<nn::OptimizableWeights>();
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
  std::string identifier("\"polymorphic_name\": \"nn::RMSprop\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}