# The cospanmdao examples, standalone like the tests: this finds the INSTALLED
# library (find_package(cospanmdao)) and is not part of the root project. Built by
# pixi-build as the package `cospanmdao-examples`; run with `pixi run gemver-chain`.
#
# examples/gemver-chain/cublas is a separate, standalone CUDA project (built in
# the `cublas` pixi environment) and is not referenced from here.
cmake_minimum_required(VERSION 3.24)
project(cospanmdao-examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AdaptiveCpp's exported acpp-rt links Threads::Threads and does not find it
# itself, so Threads comes first.
find_package(Threads REQUIRED)
find_package(AdaptiveCpp REQUIRED)
find_package(cospanmdao REQUIRED)

add_executable(gemver-chain gemver-chain/main.cpp)
target_include_directories(gemver-chain PRIVATE gemver-chain)
target_link_libraries(gemver-chain PRIVATE naga::cospan)
# Header-only library: the kernels are emitted by this translation unit.
add_sycl_to_target(TARGET gemver-chain SOURCES gemver-chain/main.cpp)

install(TARGETS gemver-chain RUNTIME DESTINATION bin)
