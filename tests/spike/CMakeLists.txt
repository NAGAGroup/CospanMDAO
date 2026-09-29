cmake_minimum_required(VERSION 3.23)
project(cospan-spike LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)  # FIX 1: AdaptiveCpp's exported acpp-rt links Threads::Threads and does not find it itself
find_package(AdaptiveCpp REQUIRED)

# Variant A: placeholder + outer kernel in a shared library; stages in the consumer.
add_library(spike_core SHARED core.cpp)
target_include_directories(spike_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_sycl_to_target(TARGET spike_core SOURCES core.cpp)
add_executable(spike_a consumer.cpp)
target_link_libraries(spike_a PRIVATE spike_core)
add_sycl_to_target(TARGET spike_a SOURCES consumer.cpp)

# Variant B: everything in one translation unit.
add_executable(spike_b single_tu.cpp)
target_include_directories(spike_b PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_sycl_to_target(TARGET spike_b SOURCES single_tu.cpp)
