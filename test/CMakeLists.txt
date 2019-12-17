find_path (CATCH2_INCLUDE_DIR NAMES "catch.hpp" PATHS "../include/catch2")
include_directories(${CATCH2_INCLUDE_DIR})

file(GLOB SOURCES "src/*.cpp")
add_executable(test ${SOURCES})

target_link_libraries(test ArrayFire::af)

find_library (NN_LIB_RELEASE_DIR NAMES "nn.lib" PATHS "../build/Release")
target_link_libraries(test optimized ${NN_LIB_RELEASE_DIR})

find_library (NN_LIB_DEBUG_DIR NAMES "nn.lib" PATHS "../build/Debug")
target_link_libraries(test debug ${NN_LIB_DEBUG_DIR})