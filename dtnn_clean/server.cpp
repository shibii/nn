#include "server.hpp"

namespace dtnn {
  Server::Server(std::string target) {
    socket_ = zmq::socket_t(context_, zmq::socket_type::router);
    socket_.bind(target);
  }
  zmq::context_t& Server::get_context() {
    return context_;
  }
  zmq::socket_t& Server::get_socket() {
    return socket_;
  }
  void Server::receive(std::string &identity, std::string &type, std::string &data) {
    zmq::multipart_t msg;
    msg.recv(socket_);
    identity = std::string(msg[0].data<char>(), msg[0].size());
    type = std::string(msg[1].data<char>(), msg[1].size());
    data = std::string(msg[2].data<char>(), msg[2].size());
  }
  void Server::respond(const std::string identity, const std::string type, const std::string data) {
    zmq::multipart_t msg;
    msg.addstr(identity);
    msg.addstr(type);
    msg.addstr(data);
    msg.send(socket_);
  }
}