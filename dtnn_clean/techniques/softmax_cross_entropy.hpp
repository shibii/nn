#pragma once

#include <arrayfire.h>
#include <cereal/types/polymorphic.hpp>

#include "../cereal_archives.hpp"
#include "../loss_function.hpp"
#include "../feed.hpp"

namespace dtnn {
  class SoftmaxCrossEntropy : public LossFunction {
  public:
    SoftmaxCrossEntropy() = default;
    ~SoftmaxCrossEntropy() = default;
    af::array error(Feed &f, af::array target) override;
    af::array loss(Feed &f, af::array target) override;
    af::array output(Feed &f) override;
    template <class Archive> void serialize(Archive &ar);
  private:
    af::array softmax(const af::array &input);
  };
}
CEREAL_REGISTER_TYPE(dtnn::SoftmaxCrossEntropy);
CEREAL_REGISTER_POLYMORPHIC_RELATION(dtnn::LossFunction, dtnn::SoftmaxCrossEntropy)