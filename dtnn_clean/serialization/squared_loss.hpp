#pragma once

#include "../techniques/squared_loss.hpp"

namespace cereal {
  template<class Archive>
  void serialize(Archive & archive, dtnn::SquaredLoss & m)
  {
  }
}
CEREAL_REGISTER_TYPE(dtnn::SquaredLoss);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::LossFunction, dtnn::SquaredLoss);