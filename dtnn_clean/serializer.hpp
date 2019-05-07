#pragma once

#include <iostream>

namespace dtnn {
  class Serializer {
  public:
    template <typename OutputArchiveType, typename... ObjectTypes>
    static void serialize(std::ostream &stream, ObjectTypes &... objects) {
      OutputArchiveType oarchive(stream);
      oarchive(objects...);
    }
    template <typename InputArchiveType, typename... ObjectTypes>
    static void deserialize(std::istream &stream, ObjectTypes &... objects) {
      InputArchiveType iarchive(stream);
      iarchive(objects...);
    }
    template <typename OutputArchiveType, typename... ObjectTypes>
    static void serialize(const char *path, ObjectTypes &... objects) {
      std::ofstream stream(path, std::ios::binary);
      serialize<OutputArchiveType>(stream, objects...);
    }
    template <typename InputArchiveType, typename... ObjectTypes>
    static void deserialize(const char *path, ObjectTypes &... objects) {
      std::ifstream stream(path, std::ios::binary);
      deserialize<InputArchiveType>(stream, objects...);
    }
  };
}