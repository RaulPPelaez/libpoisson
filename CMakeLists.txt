cmake_minimum_required(VERSION 3.24)
project(libPoisson)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all-major")
endif()
# Here I can include the logic to detect CUDA to not include the CUDA subdirectory if CUDA is not available
add_subdirectory(libPoisson/solvers_cuda)
