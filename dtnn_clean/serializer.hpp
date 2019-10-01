#pragma once

#include <iostream>

namespace nn {
  class Serializer {
  public:
    template <typename... ObjectTypes>
    static void serializeJSON(std::ostream &stream, ObjectTypes &... objects) {
      serialize<cereal::JSONOutputArchive>(stream, objects...);
    }
    template <typename... ObjectTypes>
    static void deserializeJSON(std::istream &stream, ObjectTypes &... objects) {
      deserialize<cereal::JSONInputArchive>(stream, objects...);
    }
    template <typename... ObjectTypes>
    static void serializeBinary(std::ostream &stream, ObjectTypes &... objects) {
      serialize<cereal::PortableBinaryOutputArchive>(stream, objects...);
    }
    template <typename... ObjectTypes>
    static void deserializeBinary(std::istream &stream, ObjectTypes &... objects) {
      deserialize<cereal::PortableBinaryInputArchive>(stream, objects...);
    }
    template <typename... ObjectTypes>
    static void serializeXML(std::ostream &stream, ObjectTypes &... objects) {
      serialize<cereal::XMLOutputArchive>(stream, objects...);
    }
    template <typename... ObjectTypes>
    static void deserializeXML(std::istream &stream, ObjectTypes &... objects) {
      deserialize<cereal::XMLInputArchive>(stream, objects...);
    }
  private:
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