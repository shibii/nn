#pragma once

#include <arrayfire.h>

namespace nn {
  struct Feed {
    af::array signal;
    bool is_training;
  };
}