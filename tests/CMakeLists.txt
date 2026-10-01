# The cospanmdao test package, standalone like SlabKit's tests: it finds the
# INSTALLED library (find_package(cospanmdao)) and is not part of the root
# project (the root CMakeLists does not add_subdirectory(tests)). Built by
# pixi-build as the package `cospanmdao-tests`; run with `pixi run test`.
#
# tests/spike and tests/smoke are separate, standalone projects and are not
# referenced from here.
cmake_minimum_required(VERSION 3.24)
project(cospanmdao-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AdaptiveCpp's exported acpp-rt links Threads::Threads and does not find it
# itself, so Threads comes first.
find_package(Threads REQUIRED)
find_package(AdaptiveCpp REQUIRED)
find_package(cospanmdao REQUIRED)
find_package(Catch2 3 REQUIRED)

set(COSPAN_TEST_SOURCES
    test_component.cpp
    test_flavors.cpp
    test_model.cpp
    test_ragged.cpp
    test_system.cpp
)

add_executable(cospan-tests ${COSPAN_TEST_SOURCES})
target_link_libraries(cospan-tests PRIVATE Catch2::Catch2WithMain naga::cospan)
# Header-only library: the kernels are emitted by these translation units.
add_sycl_to_target(TARGET cospan-tests SOURCES ${COSPAN_TEST_SOURCES})

install(TARGETS cospan-tests RUNTIME DESTINATION bin)
