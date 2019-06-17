#include "client.hpp"

namespace dtnn {
  Client::Client(std::string target) {
    socket_ = zmq::socket_t(context_, zmq::socket_type::dealer);
    socket_.connect(target);
  }
  zmq::context_t& Client::get_context() {
    return context_;
  }
  zmq::socket_t& Client::get_socket() {
    return socket_;
  }
  void Client::receive(std::string &type, std::string &data) {
    zmq::multipart_t msg;
    msg.recv(socket_);
    type = std::string(msg[0].data<char>(), msg[0].size());
    data = std::string(msg[1].data<char>(), msg[1].size());
  }
  void Client::send(const std::string type, const std::string data) {
    zmq::multipart_t msg;
    msg.addstr(type);
    msg.addstr(data);
    msg.send(socket_);
  }
}