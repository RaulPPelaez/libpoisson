cmake_minimum_required(VERSION 3.24)
project(libPoisson)
set(CMAKE_CUDA_ARCHITECTURES 89)
# Here I can include the logic to detect CUDA to not include the CUDA subdirectory if CUDA is not available
add_subdirectory(libPoisson/solvers_cuda)
