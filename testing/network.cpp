#include <sstream>
#include "catch.hpp"

#include "dtnn.hpp"

#include "util.hpp"

TEST_CASE("network serializes", "[network]") {
  using namespace dtnn;
  auto optimizer = std::make_shared<SGD>(1e-3f);
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
  std::string identifier("\"polymorphic_name\": \"dtnn::SGD\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
  identifier = ("\"polymorphic_name\": \"dtnn::FullyConnected\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
  identifier = ("\"polymorphic_name\": \"dtnn::Logistic\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
  identifier = ("\"polymorphic_name\": \"dtnn::SquaredLoss\"");
  REQUIRE(ostream.str().find(identifier) != std::string::npos);
}