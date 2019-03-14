#pragma once

#include <arrayfire.h>

namespace dtnn {
  struct Feed {
    af::array signal;
  };
}