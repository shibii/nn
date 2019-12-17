find_path (CATCH2_INCLUDE_DIR NAMES "catch.hpp" PATHS "../include/catch2")
include_directories(${CATCH2_INCLUDE_DIR})

file(GLOB SOURCES "src/*.cpp")
add_executable(example ${SOURCES})

target_link_libraries(example ArrayFire::af)

find_library (NN_LIB_RELEASE_DIR NAMES "nn.lib" PATHS "../build/Release")
target_link_libraries(example optimized ${NN_LIB_RELEASE_DIR})

find_library (NN_LIB_DEBUG_DIR NAMES "nn.lib" PATHS "../build/Debug")
target_link_libraries(example debug ${NN_LIB_DEBUG_DIR})