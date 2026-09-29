cmake_minimum_required(VERSION 3.23)
project(cospan-smoke LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(AdaptiveCpp REQUIRED)

# The library is header-only and not installed in the dev environment, so the
# smoke check builds it from the source tree (top-level CMakeLists.txt).
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/cospanmdao)

add_executable(smoke smoke.cpp)
target_link_libraries(smoke PRIVATE naga::cospan)
add_sycl_to_target(TARGET smoke SOURCES smoke.cpp)
