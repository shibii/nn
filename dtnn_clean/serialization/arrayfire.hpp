#pragma once

#include <arrayfire.h>
#include <cereal/types/vector.hpp>

namespace cereal {
  template <class Archive>
  void save(Archive& archive, af::array const& a) {
    std::vector<dim_t> dims{ a.dims(0), a.dims(1), a.dims(2), a.dims(3) };
    std::vector<float> elem(a.elements());
    a.host(elem.data());
    archive(dims, elem);
  }

  template <class Archive>
  void load(Archive& archive, af::array& a) {
    std::vector<dim_t> dims;
    std::vector<float> elem;
    archive(dims, elem);
    if (!elem.empty())
      a = af::array(af::dim4(dims[0], dims[1], dims[2], dims[3]),
        elem.data());
  }
}