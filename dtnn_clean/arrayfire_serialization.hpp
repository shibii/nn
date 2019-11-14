#pragma once

#include <arrayfire.h>
#include <cereal/types/vector.hpp>
#include "arrayfire_util.hpp"

namespace cereal {
template <class Archive>
void save(Archive& archive, af::array const& a) {
  std::vector<dim_t> dims{a.dims(0), a.dims(1), a.dims(2), a.dims(3)};
  auto elements = nn::util::vectorize(a);
  archive(dims, elements);
}

template <class Archive>
void load(Archive& archive, af::array& a) {
  std::vector<dim_t> dims;
  std::vector<float> elem;
  archive(dims, elem);
  if (!elem.empty())
    a = af::array(af::dim4(dims[0], dims[1], dims[2], dims[3]), elem.data());
}

template <class Archive>
void save(Archive& archive, af::dim4 const& d) {
  std::vector<dim_t> dims{d[0], d[1], d[2], d[3]};
  archive(dims);
}

template <class Archive>
void load(Archive& archive, af::dim4& d) {
  std::vector<dim_t> dims;
  archive(dims);
  d = {dims[0], dims[1], dims[2], dims[3]};
}
}  // namespace cereal