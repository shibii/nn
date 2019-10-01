#include <sstream>
#include "catch.hpp"

#include "nn.hpp"

#include "util.hpp"

TEST_CASE("network serializes", "[network]") {
  using namespace nn;
  auto optimizer = std::make_shared<SGD>();
  Network nn(af::dim4(1), optimizer);
  auto fc1 = std::make_shared<FullyConnected>(20);
  nn.add(fc1);
  nn.add(std::make_shared<Logistic>());
  auto loss = std::make_shared<SquaredLoss>();
  nn.add(loss);

  std::ostringstream ostream;
  {
    cereal::JSONOutputArchive oarchive(ostream);
    oarchive(nn);
  }
  std::string identifier("\"polymorphic_name\": \"nn::SGD\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
  identifier = ("\"polymorphic_name\": \"nn::FullyConnected\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
  identifier = ("\"polymorphic_name\": \"nn::Logistic\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
  identifier = ("\"polymorphic_name\": \"nn::SquaredLoss\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}