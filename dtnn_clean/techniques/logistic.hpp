#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../weightless_stage.hpp"
#include "../feed.hpp"

namespace dtnn {
  class Logistic : public WeightlessStage {
  public:
    ~Logistic() = default;
    Logistic() = default;
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    template <class Archive> void serialize(Archive &ar);

  private:
    af::array activation_;
  };
}
CEREAL_REGISTER_TYPE(dtnn::Logistic);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::PropagationStage, dtnn::Logistic)