#pragma once

#define ZMQ_STATIC
#include <zmq.hpp>

namespace dtnn {
  class Endpoint {
  public:
    virtual zmq::context_t& get_context() = 0;
    virtual zmq::socket_t& get_socket() = 0;
  };
}