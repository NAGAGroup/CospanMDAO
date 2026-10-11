cmake_minimum_required(VERSION 3.24)
project(getting-started LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(AdaptiveCpp REQUIRED)
find_package(cospanmdao REQUIRED)

add_executable(getting-started main.cpp)
target_link_libraries(getting-started PRIVATE naga::cospan)
add_sycl_to_target(TARGET getting-started SOURCES main.cpp)

install(TARGETS getting-started)
