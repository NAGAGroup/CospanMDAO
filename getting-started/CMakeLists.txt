cmake_minimum_required(VERSION 3.24)
project(getting-started LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The Windows toolchain's activation puts -std=c++17 in CXXFLAGS, hence in CMAKE_CXX_FLAGS, and acpp refuses two -std options
# ("Multiple c++ standards defined"). CMake adds the one CMAKE_CXX_STANDARD asks for, so drop the inherited one.
string(REGEX REPLACE "(^| )-std=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(AdaptiveCpp REQUIRED)
find_package(cospanmdao REQUIRED)

add_executable(getting-started main.cpp)
target_link_libraries(getting-started PRIVATE naga::cospan)
add_sycl_to_target(TARGET getting-started SOURCES main.cpp)

install(TARGETS getting-started)
