#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include <arrayfire.h>

int main(int argc, char* argv[]) {
  std::cout << "available backends: " << af::getBackendCount() << std::endl;

  int backends = af::getAvailableBackends();
  bool cpu = backends & AF_BACKEND_CPU;
  bool cuda = backends & AF_BACKEND_CUDA;
  bool opencl = backends & AF_BACKEND_OPENCL;

  Catch::Session session;

  if (cpu) {
    std::cout << "testing cpu backend" << std::endl;
    af::setBackend(AF_BACKEND_CPU);
    af::info();
    session.run(argc, argv);
  }

  if (opencl) {
    std::cout << "testing opencl backend" << std::endl;
    af::setBackend(AF_BACKEND_OPENCL);
    af::info();
    session.run(argc, argv);
  }

  if (cuda) {
    std::cout << "testing cuda backend" << std::endl;
    af::setBackend(AF_BACKEND_CUDA);
    af::info();
    session.run(argc, argv);
  }

  return 0;
}