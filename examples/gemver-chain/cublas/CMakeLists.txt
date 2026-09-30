# The cuBLAS reference of the gemver-chain example: a standalone CUDA project,
# built by the `cublas` pixi environment (pixi run -e cublas build-cublas) and NOT
# part of the cospanmdao-examples package. Release only, never -g.
cmake_minimum_required(VERSION 3.24)
project(gemver-chain-cublas LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gemver-chain-cublas main.cu)
target_include_directories(gemver-chain-cublas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(gemver-chain-cublas PRIVATE CUDA::cublas CUDA::cudart)
else()
    # CUDAToolkit not found in the environment: link cublas and cudart from $CONDA_PREFIX.
    find_library(GC_CUBLAS cublas HINTS $ENV{CONDA_PREFIX}/lib $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib REQUIRED)
    find_library(GC_CUDART cudart HINTS $ENV{CONDA_PREFIX}/lib $ENV{CONDA_PREFIX}/targets/x86_64-linux/lib REQUIRED)
    target_include_directories(gemver-chain-cublas PRIVATE $ENV{CONDA_PREFIX}/include
                               $ENV{CONDA_PREFIX}/targets/x86_64-linux/include)
    target_link_libraries(gemver-chain-cublas PRIVATE ${GC_CUBLAS} ${GC_CUDART})
endif()
