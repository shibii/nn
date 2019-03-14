#pragma once

#include "../wb.hpp";

namespace cereal {
  template<class Archive>
  void serialize(Archive & archive, dtnn::wb & m)
  {
    archive(m.w, m.b);
  }
}