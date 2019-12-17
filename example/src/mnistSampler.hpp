#pragma once

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

bool parseMnist(const std::string &imagePath, const std::string &labelPath,
                std::vector<float> &inputData, std::vector<float> &targetData) {
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

    inputData.resize(n * r * c);
    for (int v = 0; v < inputData.size(); v++) {
      uint8_t i;
      f.read((char *)&i, sizeof(uint8_t));
      inputData[v] = (float)i / 255.f;
    }
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

    targetData.resize(n * 10, 0.f);
    for (int v = 0; v < n; v++) {
      uint8_t i;
      f.read((char *)&i, sizeof(uint8_t));

      targetData[v * 10 + i] = 1.f;
    }
    f.close();
    return true;
  } else {
    return false;
  }
}
};  // namespace MNIST
