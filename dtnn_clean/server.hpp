#pragma once

#include "endpoint.hpp"
#include "zmq_addon.hpp"

namespace dtnn {
  class Server : public Endpoint {
  public:
    Server(std::string target);
    zmq::context_t& get_context() override;
    zmq::socket_t& get_socket() override;
    void receive(std::string &identity, std::string &type, std::string &data);
    void respond(const std::string identity, const std::string type, const std::string data);
  private:
    zmq::context_t context_;
    zmq::socket_t socket_;
  };
}