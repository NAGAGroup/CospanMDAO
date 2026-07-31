cmake_minimum_required(VERSION 3.23)
project(
    cospanmdao
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "CospanMDAO: GPU-native differentiable modeling and optimization framework"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(AdaptiveCpp REQUIRED)

# ------------------------------------------------------------------------------
# Package manager
# ------------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(NAGA_BUILD_TESTS "Build unit tests" ON)
option(NAGA_BUILD_EXAMPLES "Build example apps" ON)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
add_subdirectory(ext/slabkit)
set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "Skip Boost requirement" FORCE)
CPMAddPackage("gh:USCiLab/cereal@1.3.2")

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------
add_library(naga STATIC
    src/naga/core/context_core.cpp
    src/naga/core/context_io.cpp
    src/naga/core/context_execution.cpp
    src/naga/core/context_finalization.cpp
    src/naga/core/context_graph.cpp
    src/naga/core/context_serialization.cpp
    src/naga/core/equilibrium_executor.cpp
    src/naga/core/system.cpp
    src/naga/core/executor.cpp
    src/naga/core/component.cpp
)
target_include_directories(naga PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(naga PUBLIC slabkit cereal::cereal)
add_sycl_to_target(TARGET naga SOURCES
    src/naga/core/equilibrium_executor.cpp
)

# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------
if(NAGA_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
if(NAGA_BUILD_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()
