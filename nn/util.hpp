#pragma once

#include <ctime>
#include <arrayfire.h>

namespace nn {
enum BACKEND { CPU, OPENCL, CUDA };
static void set_backend(BACKEND backend) {
  switch (backend) {
    case nn::CPU:
      af::setBackend(AF_BACKEND_CPU);
      break;
    case nn::OPENCL:
      af::setBackend(AF_BACKEND_OPENCL);
      break;
    case nn::CUDA:
      af::setBackend(AF_BACKEND_CUDA);
      break;
    default:
      af::setSeed(time(NULL));
      break;
  }
}
static void seed(unsigned long long seed) { af::setSeed(seed); }
static void info() { af::info(); }
}