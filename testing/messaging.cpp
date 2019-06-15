#include "catch.hpp"

#include "server.hpp"
#include "client.hpp"

TEST_CASE("send and reveice messages between endpoints", "[messaging]") {
  dtnn::Server server("tcp://127.0.0.1:5555");
  dtnn::Client client("tcp://127.0.0.1:5555");

  client.send("test", "data");
  std::string identity;
  std::string type;
  std::string data;
  server.receive(identity, type, data);
  REQUIRE(type == "test");
  REQUIRE(data == "data");
}