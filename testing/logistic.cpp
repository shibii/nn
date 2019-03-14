#include "catch.hpp"
#include <dtnn.hpp>
#include "util.hpp"
#include <sstream>

TEST_CASE("logistic forward", "[logistic]") {
  float hostinput[] = { -1e+20f, -1.f, 0.f, 1.f, 1e+20f };
  float hostexpected[] = { 0.f, .2689f, .5f, .7310f, 1.f};
  af::array input = af::array(af::dim4(5), hostinput);
  af::array expected = af::array(af::dim4(5), hostexpected);

  dtnn::Feed f;
  f.signal = input;
  auto logistic = dtnn::Logistic();
  logistic.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));
}

TEST_CASE("logistic serializes", "[logistic]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto af = std::make_shared<dtnn::Logistic>(dtnn::Logistic());
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::Logistic\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}