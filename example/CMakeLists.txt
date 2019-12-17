file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/example/src/*.cpp)
add_executable(example ${SOURCES})

target_link_libraries(example ArrayFire::af)
target_link_libraries(example optimized ${PROJECT_SOURCE_DIR}/build/Release/nn.lib)
target_link_libraries(example debug ${PROJECT_SOURCE_DIR}/build/Debug/nn.lib)