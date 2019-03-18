#pragma once

#include <arrayfire.h>

#include "../serialization.hpp"
#include "../weightless_stage.hpp"
#include "../feed.hpp"

namespace dtnn {
  class Logistic : public WeightlessStage {
  public:
    ~Logistic() = default;
    Logistic() = default;
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    template<class Archive> void serialize(Archive & archive);

  private:
    af::array activation_;
  };
}
CEREAL_REGISTER_TYPE(dtnn::Logistic);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::PropagationStage, dtnn::Logistic);