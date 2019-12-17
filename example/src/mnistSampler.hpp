#pragma once

#include <arrayfire.h>

#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>

namespace MNIST {
int32_t swapEndianness(int32_t i) {
  return (((i & 0xFF) << 24) | ((i & 0xFF00) << 8) | ((i & 0xFF0000) >> 8) |
          ((i & 0xFF000000) >> 24));
}

bool parseMnist(const char *imagePath, const char *labelPath,
                af::array &inputData, af::array &targetData) {
  // inputs
  std::ifstream f;
  f.open(imagePath, std::ios::in | std::ios::binary);
  if (f.is_open()) {
    int32_t mn;
    f.read((char *)&mn, sizeof mn);
    mn = swapEndianness(mn);
    assert(mn == 2051);

    int32_t n;
    f.read((char *)&n, sizeof n);
    n = swapEndianness(n);

    int32_t r;
    f.read((char *)&r, sizeof r);
    r = swapEndianness(r);

    int32_t c;
    f.read((char *)&c, sizeof c);
    c = swapEndianness(c);

    std::vector<float> inputsVec(n * r * c);

    for (int v = 0; v < inputsVec.size(); v++) {
      uint8_t i;
      f.read((char *)&i, sizeof(uint8_t));
      inputsVec[v] = (float)i / 255.f;
    }
    inputData = af::array(af::dim4(r, c, 1, n), inputsVec.data());
    f.close();

    // targets
    f.open(labelPath, std::ios::in | std::ios::binary);

    mn;
    f.read((char *)&mn, sizeof mn);
    mn = swapEndianness(mn);
    assert(mn == 2049);

    n;
    f.read((char *)&n, sizeof n);
    n = swapEndianness(n);

    std::vector<float> targetsVec(n * 10, 0.f);

    for (int v = 0; v < n; v++) {
      uint8_t i;
      f.read((char *)&i, sizeof(uint8_t));

      targetsVec[v * 10 + i] = 1.f;
    }
    targetData = af::array(af::dim4(10, 1, 1, n), targetsVec.data());
    f.close();
    return true;
  } else {
    return false;
  }
}
};  // namespace MNIST
