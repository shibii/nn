#include <sstream>
#include "catch.hpp"

#include "techniques/convolutional.hpp"
#include "feed.hpp"

#include "util.hpp"

TEST_CASE("convolutional", "[convolutional]") {
  float hweights[] = { 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1, 2, 2, 1, 1, 0};
  float hbias[] = { -3, -4 };

  dtnn::Feed f;
  float hinput[] = { 1,2,1,2,0,0,1,1,1,0,2,1,1,0,0,1,1,1,
    1,1,1,0,0,0,1,1,1,1,2,0,2,0,1,0,1,0,
    0,2,0,2,0,0,1,1,2,1,2,0,0,2,0,1,1,1 };
  f.signal = af::array(af::dim4(3, 3, 2, 3), hinput);

  auto conv = dtnn::Convolutional(2, 2, 1, 1, 0, 0, 2);
  auto param = conv.init(f.signal.dims());
  param->weights = {
    af::array(af::dim4(8, 2), hweights),
    af::array(af::dim4(1, 2), hbias)
  };
  param->gradient = {
    af::constant(0.f, af::dim4(8, 2)),
    af::constant(0.f, af::dim4(1, 2))
  };

  conv.forward(f);

  float hexpected[] = { 6,6,3,-1,3,2,2,0,5,3,3,1,3,1,3,1,7,3,5,4,4,2,2,6 };
  af::array expected = af::array(af::dim4(2, 2, 2, 3), hexpected);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expected));
  REQUIRE(util::approx(f.signal, expected));

  float hexpectedgw[] = { 5,8,9,4,10,10,9,7,5,7,8,6,15,5,3,8 };
  af::array expectedgw = af::array(af::dim4(8, 2), hexpectedgw);
  float hexpectedgb[] = { 9, 9 };
  af::array expectedgb = af::array(af::dim4(1, 2), hexpectedgb);

  float herror[] = { 1,0,0,1,1,1,0,0,2,0,1,0,0,2,2,0,2,1,0,1,1,0,0,2 };
  f.signal = af::array(af::dim4(2, 2, 2, 3), herror);
  conv.backward(f);
  REQUIRE(util::isnumber(param->gradient.w));
  REQUIRE(util::isnumber(param->gradient.b));
  REQUIRE(util::approx(param->gradient.w, expectedgw));
  REQUIRE(util::approx(param->gradient.b, expectedgb));

  float hexpectederror[] = { 1,2,1,2,5,3,0,1,1,4,5,1,1,3,2,0,0,0,
    2,2,2,3,7,4,3,5,0,4,8,2,6,6,0,2,0,0,
    2,4,1,3,6,4,0,3,5,6,7,2,1,6,4,0,2,0 };
  af::array expectederror = af::array(af::dim4(3, 3, 2, 3), hexpectederror);

  REQUIRE(util::isnumber(f.signal));
  REQUIRE(util::samedim(f.signal, expectederror));
  REQUIRE(util::approx(f.signal, expectederror));
}

TEST_CASE("convolutional serializes", "[convolutional]") {
  std::vector<std::shared_ptr<dtnn::PropagationStage>> stages;
  auto weights = std::make_shared<dtnn::wb>(dtnn::wb());
  auto conv = std::make_shared<dtnn::Convolutional>(dtnn::Convolutional(2,2,1,1,1,1,8));

  dtnn::Feed f;
  f.signal = af::constant(0.f, af::dim4(2, 1, 1, 2));
  conv->init(f.signal.dims());

  stages.push_back(conv);
  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(stages);
  }
  std::string identifier("\"polymorphic_name\": \"dtnn::Convolutional\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}