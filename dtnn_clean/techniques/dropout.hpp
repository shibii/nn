#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../weightless_stage.hpp"
#include "../feed.hpp"

namespace nn {
  class Dropout : public WeightlessStage {
  public:
    ~Dropout() = default;
    Dropout(float pass_probability = 0.5f);
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    template <class Archive> void serialize(Archive &ar);

    float pass_probability_;
  private:
    af::array passmask_;
  };
}
CEREAL_REGISTER_TYPE(nn::Dropout);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::Dropout)