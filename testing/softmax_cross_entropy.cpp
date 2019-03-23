#include <sstream>
#include "catch.hpp"

#include "techniques/softmax_cross_entropy.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("softmax cross entropy", "[softmax cross entropy]") {

  float h_input[] = { -1e+20f, -1.f, 0.f, 1.f, 1e+20f, 1e+20f};
  float h_target[] = { 0, 0, 0, 1, 0, 0 };
  float h_output[] = { 0, 0, 0, 0, .5, .5 };
  float h_error[] = { 0, 0, 0, -1, .5, .5 };
  float h_loss[] = { 0, 0, 0, 1e+20f, 0, 0 };
  af::array input = af::array(af::dim4(6), h_input);
  af::array target = af::array(af::dim4(6), h_target);
  af::array output = af::array(af::dim4(6), h_output);
  af::array error = af::array(af::dim4(6), h_error);
  af::array loss = af::array(af::dim4(6), h_loss);

  dtnn::Feed f;
  f.signal = input;
  auto lf = dtnn::SoftmaxCrossEntropy();

  REQUIRE(util::approx(output, lf.output(f)));
  REQUIRE(util::approx(error, lf.error(f, target)));
  REQUIRE(util::approx(loss, lf.loss(f, target)));
}

TEST_CASE("softmax cross entropy serializes", "[softmax cross entropy]") {
  std::shared_ptr<dtnn::LossFunction> loss;
  loss = std::make_shared<dtnn::SoftmaxCrossEntropy>(dtnn::SoftmaxCrossEntropy());
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(loss);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::SoftmaxCrossEntropy\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}