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

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(AdaptiveCpp REQUIRED)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(NAGA_BUILD_TESTS "Build unit tests" ON)
option(NAGA_BUILD_EXAMPLES "Build example apps" ON)

# ------------------------------------------------------------------------------
# Dependencies — provided by pixi as source packages (SlabKit, cereal).
# ------------------------------------------------------------------------------
find_package(slabkit CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)

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
add_library(cospanmdao::naga ALIAS naga)
target_include_directories(naga PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(naga PUBLIC slabkit::slabkit cereal::cereal)
add_sycl_to_target(TARGET naga SOURCES
    src/naga/core/equilibrium_executor.cpp
)

# ------------------------------------------------------------------------------
# Install / export — consumers do find_package(cospanmdao CONFIG) and link
# cospanmdao::naga.
# ------------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS naga EXPORT cospanmdaoTargets)

install(EXPORT cospanmdaoTargets
    FILE cospanmdaoTargets.cmake
    NAMESPACE cospanmdao::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cospanmdao
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cospanmdaoConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cospanmdao
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cospanmdao
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
