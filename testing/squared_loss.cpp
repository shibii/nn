#include <sstream>
#include "catch.hpp"

#include "techniques/squared_loss.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("squared loss error", "[squared loss]") {
  float h_input[] = { 500, 0, -100 };
  float h_target[] = { 455, -100, 100 };
  float h_output[] = { 500, 0, -100 };
  float h_error[] = { 45, 100, -200 };
  float h_loss[] = { 1012.5, 5000, 20000 };
  af::array input = af::array(af::dim4(3), h_input);
  af::array target = af::array(af::dim4(3), h_target);
  af::array output = af::array(af::dim4(3), h_output);
  af::array error = af::array(af::dim4(3), h_error);
  af::array loss = af::array(af::dim4(3), h_loss);

  dtnn::Feed f;
  f.signal = input;
  auto lf = dtnn::SquaredLoss();

  REQUIRE(util::approx(output, lf.output(f)));
  REQUIRE(util::approx(error, lf.error(f, target)));
  REQUIRE(util::approx(loss, lf.loss(f, target)));
}

TEST_CASE("squared loss serializes", "[squared loss]") {
  std::shared_ptr<dtnn::LossFunction> loss;
  loss = std::make_shared<dtnn::SquaredLoss>(dtnn::SquaredLoss());
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(loss);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::SquaredLoss\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}