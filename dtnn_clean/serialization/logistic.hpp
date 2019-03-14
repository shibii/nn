#pragma once

#include "../techniques/logistic.hpp"

namespace cereal {
  template<class Archive>
  void serialize(Archive & archive, dtnn::Logistic & m)
  {
  }
}
CEREAL_REGISTER_TYPE(dtnn::Logistic);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::PropagationStage, dtnn::Logistic);