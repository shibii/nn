find_path (CATCH2_INCLUDE_DIR NAMES "catch.hpp" PATHS ${PROJECT_SOURCE_DIR}/include/catch2)
include_directories(${CATCH2_INCLUDE_DIR})

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/test/src/*.cpp)
add_executable(test ${SOURCES})

target_link_libraries(test ArrayFire::af)
target_link_libraries(test optimized ${PROJECT_SOURCE_DIR}/build/Release/nn.lib)
target_link_libraries(test debug ${PROJECT_SOURCE_DIR}/build/Debug/nn.lib)