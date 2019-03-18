#include <sstream>
#include "catch.hpp"

#include "techniques/squared_loss.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("squared loss error", "[squared loss]") {
  float hostinput[] = { 1.f, 1.f, 2.f };
  float hosttarget[] = { 1.f, 2.f, -1.f };
  float hostexpected[] = { 0.f, -1.f, 3.f };
  af::array input = af::array(af::dim4(3), hostinput);
  af::array target = af::array(af::dim4(3), hosttarget);
  af::array expected = af::array(af::dim4(3), hostexpected);

  dtnn::Feed f;
  f.signal = input;
  auto lf = dtnn::SquaredLoss();
  af::array error = lf.error(f, target);

  REQUIRE(util::isnumber(error));
  REQUIRE(util::samedim(error, expected));
  REQUIRE(util::approx(error, expected));
}

TEST_CASE("squared loss serializes", "[squared loss]") {
  std::vector<std::shared_ptr<dtnn::LossFunction>> polymloss;
  auto loss = std::make_shared<dtnn::SquaredLoss>(dtnn::SquaredLoss());
  polymloss.push_back(loss);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(polymloss);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::SquaredLoss\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}