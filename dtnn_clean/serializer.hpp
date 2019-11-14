#pragma once

#include <iostream>

#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/portable_binary.hpp>

namespace nn {
  class Serializer {
  public:
    template <typename... ObjectTypes>
    static void serializeJSON(std::ostream &stream, ObjectTypes &... objects) {
      cereal::JSONOutputArchive oarchive(stream);
      oarchive(objects...);
    }
    template <typename... ObjectTypes>
    static void deserializeJSON(std::istream &stream, ObjectTypes &... objects) {
      cereal::JSONInputArchive iarchive(stream);
      iarchive(objects...);
    }
    template <typename... ObjectTypes>
    static void serializeBinary(std::ostream &stream, ObjectTypes &... objects) {
      cereal::PortableBinaryOutputArchive oarchive(stream);
      oarchive(objects...);
    }
    template <typename... ObjectTypes>
    static void deserializeBinary(std::istream &stream, ObjectTypes &... objects) {
      cereal::PortableBinaryInputArchive iarchive(stream);
      iarchive(objects...);
    }
    template <typename... ObjectTypes>
    static void serializeXML(std::ostream &stream, ObjectTypes &... objects) {
      cereal::XMLOutputArchive oarchive(stream);
      oarchive(objects...);
    }
    template <typename... ObjectTypes>
    static void deserializeXML(std::istream &stream, ObjectTypes &... objects) {
      cereal::XMLInputArchive iarchive(stream);
      iarchive(objects...);
    }
  };
}