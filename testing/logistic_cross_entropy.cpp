#include <sstream>
#include "catch.hpp"

#include "techniques/logistic_cross_entropy.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("logistic cross entropy", "[logistic cross entropy]") {
  float h_input[] = { -1e+20f, -1.f, 0.f, 1.f, 1e+20f, -1e+20f, -1.f, 0.f, 1.f, 1e+20f };
  float h_target[] = { 1.f, 1.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f , 0.f , 0.f };
  float h_output[] = { 0, .2689, .5, .7311, 1, 0, .2689, .5, .7311, 1 };
  float h_error[] = { -1, -.7311, -.5, -.2690, 0, 0, .2689, .5, .7311, 1 };
  float h_loss[] = { 1e+20f, 1.3133f, .6931, .3133, 0, 0, 0.3133, .6931, 1.3133, 1e+20f };
  af::array input = af::array(af::dim4(10), h_input);
  af::array target = af::array(af::dim4(10), h_target);
  af::array output = af::array(af::dim4(10), h_output);
  af::array error = af::array(af::dim4(10), h_error);
  af::array loss = af::array(af::dim4(10), h_loss);

  dtnn::Feed f;
  f.signal = input;
  auto lf = dtnn::LogisticCrossEntropy();

  REQUIRE(util::approx(output, lf.output(f)));
  REQUIRE(util::approx(error, lf.error(f, target)));
  REQUIRE(util::approx(loss, lf.loss(f, target)));
}

TEST_CASE("logistic cross entropy serializes", "[logistic cross entropy]") {
  std::shared_ptr<dtnn::LossFunction> loss;
  loss = std::make_shared<dtnn::LogisticCrossEntropy>(dtnn::LogisticCrossEntropy());
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(loss);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::LogisticCrossEntropy\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}