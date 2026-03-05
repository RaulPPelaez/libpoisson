cmake_minimum_required(VERSION 3.24)
project(libPoisson)
enable_language(CUDA)

option(DOUBLEPRECISION "Build with double precision" OFF)
# For the future when I test in cpp
# option(INSTALL_TESTS "Install tests" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# For the future when I add debug mode
#if(CMAKE_BUILD_TYPE MATCHES "Debug")
#    add_compile_definitions(PUBLIC MAXLOGLEVEL=15)
#else()
#    add_compile_definitions(PUBLIC MAXLOGLEVEL=3)
#endif()
add_compile_definitions(PUBLIC MAXLOGLEVEL=5)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
# Set CUDA archs so all supported GPUs are covered
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all")
endif()
IF (CMAKE_BUILD_TYPE MATCHES "Debug")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "-g -G -src-in-ptx -fPIC")
  else()
    set(CMAKE_CUDA_FLAGS "-g")
  endif()
 set(CMAKE_CXX_FLAGS "-g -Wall -fPIC")
ENDIF()
if(DOUBLEPRECISION) # checks if equal to 1
  add_compile_definitions(PUBLIC DOUBLE_PRECISION)
endif()
find_package(Python 3
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

## FetchConntent for dependencies, include them when needed.
#include(FetchContent)
#find_package(nanobind QUIET)
#if (NOT nanobind_FOUND)
#  FetchContent_Declare(
#    nanobind
#    GIT_REPOSITORY https://github.com/wjakob/nanobind/
#    GIT_TAG        v2.7.0
#  )
#  FetchContent_MakeAvailable(nanobind)
#endif()
#
#FetchContent_Declare(
#  uammd
#  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd/
#  GIT_TAG        v3.0.1
#  EXCLUDE_FROM_ALL
#)
#FetchContent_MakeAvailable(uammd)

## Other dependencies, include them when needed.
#set(BLA_VENDOR Generic)
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

## Project structure, include them when needed.
include_directories(${CMAKE_SOURCE_DIR}/include)
#add_subdirectory(solvers)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -extended-lambda")
endif()

## Testing in cpp, include them when needed.
#enable_testing()
#add_subdirectory(tests/cpp)
#if (INSTALL_TESTS)
#  install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share/libMobility/tests/ FILES_MATCHING PATTERN "*.py" PATTERN "*.npz")
#endif()

## FOR TESTING WITH PYTEST
# Add a test for each Python file in the tests directory
#file(GLOB TESTS tests/*.py)
#foreach(TEST ${TESTS})
#  get_filename_component(TEST_NAME ${TEST} NAME_WE)
#  add_test(NAME ${TEST_NAME} COMMAND pytest -vs ${TEST})
#endforeach()
