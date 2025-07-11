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
set(NAME libpoisson)
nanobind_add_module(
  _${NAME}
  STABLE_ABI
  python_wrapper.cu
)
message(STATUS "Include directory for common: ${PROJECT_SOURCE_DIR}/src")
target_include_directories(_${NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(_${NAME} PRIVATE uammd_wrapper)
install(TARGETS _${NAME} LIBRARY DESTINATION ${Python_SITEARCH}/${NAME})
install(FILES __init__.py DESTINATION ${Python_SITEARCH}/${NAME})
