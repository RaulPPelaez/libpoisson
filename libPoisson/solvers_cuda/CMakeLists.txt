
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
        message(STATUS "Using CMAKE_CUDA_ARCHITECTURES from environment variable: $ENV{CMAKE_CUDA_ARCHITECTURES}")
        set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
    else()
        message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to 'all-major'")
        set(CMAKE_CUDA_ARCHITECTURES "all-major")
    endif()
else()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES from CMake variable: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

enable_language(CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/" "$ENV{CONDA_PREFIX}/share/cmake/Modules")
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_compile_definitions(PUBLIC MAXLOGLEVEL=5)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

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

include(FetchContent)
find_package(nanobind QUIET)
if (NOT nanobind_FOUND)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind/
    GIT_TAG        v2.7.0
  )
  FetchContent_MakeAvailable(nanobind)
endif()
FetchContent_Declare(
  uammd
  GIT_REPOSITORY https://github.com/RaulPPelaez/UAMMD.git
  GIT_TAG        v3.1.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(uammd)

set(BLA_VENDOR Generic)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

nanobind_add_module(
    _uammd
    uammd_wrapper/uammd_wrapper.cu
)
uammd_setup_target(_uammd)
target_link_libraries(_uammd PRIVATE cblas)

install(TARGETS _uammd LIBRARY DESTINATION solvers_cuda)
