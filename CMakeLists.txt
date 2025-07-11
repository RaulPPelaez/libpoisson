cmake_minimum_required(VERSION 3.24)
project(charged_ball)
enable_language(CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
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

add_subdirectory(src)
