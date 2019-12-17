cmake_minimum_required (VERSION 3.0.2)
project (NNLIBRARY)

add_compile_options(-bigobj)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
IF (WIN32)
	add_compile_definitions(NOMINMAX)
ENDIF()

find_package(ArrayFire)
find_path (ARRAYFIRE_INCLUDE_DIR NAMES "arrayfire.h" PATHS "./include/arrayfire")
include_directories(${ARRAYFIRE_INCLUDE_DIR})

find_path (CEREAL_INCLUDE_DIR NAMES "cereal.hpp" PATHS "./include/cereal")
include_directories(${CEREAL_INCLUDE_DIR}/..)

find_path (NN_INCLUDE_DIR NAMES "nn.hpp" PATHS "./include/nn")
include_directories(${NN_INCLUDE_DIR} ${NN_INCLUDE_DIR}/techniques)

file(GLOB SOURCES "./src/*.cpp" "./src/techniques/*.cpp")
add_library(nn ${SOURCES})

add_subdirectory(test)
add_subdirectory(example)