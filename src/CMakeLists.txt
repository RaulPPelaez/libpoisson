include(FetchContent)
FetchContent_Declare(
  uammd
  GIT_REPOSITORY https://github.com/RaulPPelaez/uammd/
  GIT_TAG        v2.8.6
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(uammd)

set(BLA_VENDOR Generic)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -extended-lambda")
endif()
option(MAXLOGLEVEL "Set maximum log level for uammd" 1)
add_compile_definitions(MAXLOGLEVEL=${MAXLOGLEVEL})
set(NAME uammd_wrapper)
add_library(${NAME} uammd_wrapper.cu)
set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
uammd_setup_target(${NAME})

install(TARGETS ${NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

add_subdirectory(libpoisson)
