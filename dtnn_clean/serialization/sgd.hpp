#pragma once

#include "../techniques/sgd.hpp";

namespace cereal {
  template<class Archive>
  void serialize(Archive & archive, dtnn::SGD & m)
  {
    archive(m.learningrate_, m.params_);
  }
}
CEREAL_REGISTER_TYPE(dtnn::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::Optimizer, dtnn::SGD);