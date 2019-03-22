#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../weightless_stage.hpp"
#include "../feed.hpp"

namespace dtnn {
  class Tanh : public WeightlessStage {
  public:
    ~Tanh() = default;
    Tanh() = default;
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    template <class Archive> void serialize(Archive &ar);

  private:
    af::array activation_;
  };
}
CEREAL_REGISTER_TYPE(dtnn::Tanh);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::PropagationStage, dtnn::Tanh)