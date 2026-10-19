cmake_minimum_required(VERSION 3.23)
project(cospanmdao VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AdaptiveCpp's exported acpp-rt links Threads::Threads and does not find it
# itself (found by the tests/spike project), so Threads comes first.
find_package(Threads REQUIRED)
find_package(AdaptiveCpp REQUIRED)
find_package(slabkit REQUIRED)

# Header-only (decision D2/D4): placeholders, outer kernels and stages must all
# be emitted by the consumer's translation unit, so nothing is compiled here.
# Consumers call add_sycl_to_target on their own targets.
add_library(cospanmdao INTERFACE)
add_library(naga::cospan ALIAS cospanmdao)
set_target_properties(cospanmdao PROPERTIES EXPORT_NAME cospan)
target_include_directories(cospanmdao INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cospanmdao INTERFACE naga::slabkit)
target_compile_features(cospanmdao INTERFACE cxx_std_20)

include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS cospanmdao EXPORT cospanmdaoTargets)
install(EXPORT cospanmdaoTargets
    NAMESPACE naga::
    DESTINATION lib/cmake/cospanmdao)

configure_package_config_file(
    cmake/cospanmdaoConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfig.cmake
    INSTALL_DESTINATION lib/cmake/cospanmdao)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cospanmdaoConfigVersion.cmake
    DESTINATION lib/cmake/cospanmdao)
