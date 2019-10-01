#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../weightless_stage.hpp"
#include "../feed.hpp"

namespace nn {
  class LeakyReLU : public WeightlessStage {
  public:
    ~LeakyReLU() = default;
    LeakyReLU() = default;
    LeakyReLU(float leak);
    void forward(Feed &f) override;
    void backward(Feed &f) override;
    template <class Archive> void serialize(Archive &ar);

  private:
    af::array input_;
    float leak_;
  };
}
CEREAL_REGISTER_TYPE(nn::LeakyReLU);
CEREAL_REGISTER_POLYMORPHIC_RELATION(nn::PropagationStage, nn::LeakyReLU)