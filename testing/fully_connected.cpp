#include "catch.hpp"
#include <dtnn.hpp>
#include "util.hpp"
#include <sstream>

TEST_CASE("fullyconnected forward", "[fullyconnected]") {
  float hostweights[] = { .1, 1, .2, 2, -.1, -1, -.4, -2,
                          1, .3, 4, .4, 1, -.2, -2, .5 };
  float hostbias[] = { 1.5f, 2.5f };
  float hostinput[] = { 0.f, 1.f, -1.f, -2.f, 2.f, -3.f, 3.f, 4.f};
  float hostexpected[] = { -12.4f, 10.3f };
  af::array input = af::array(af::dim4(2,2,2), hostinput);
  af::array expected = af::array(af::dim4(2), hostexpected);

  dtnn::Feed f;
  f.signal = input;
  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  weights->w = af::array(af::dim4(2, 8), hostweights);
  weights->b = af::array(af::dim4(2), hostbias);
  auto fc = dtnn::FullyConnected(weights);
  fc.forward(f);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));
}

TEST_CASE("fullyconnected serializes", "[fullyconnected]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  auto af = std::make_shared<dtnn::FullyConnected>(dtnn::FullyConnected(weights));
  stages.push_back(af);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::FullyConnected\"");
  std::cout << ostream.str();
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}