#pragma once

#include "../techniques/fully_connected.hpp"

namespace cereal {
  template<class Archive>
  void serialize(Archive & archive, dtnn::FullyConnected & m)
  {
    archive(m.weights_);
  }
}
CEREAL_REGISTER_TYPE(dtnn::FullyConnected);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::PropagationStage, dtnn::FullyConnected);