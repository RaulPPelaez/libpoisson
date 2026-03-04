#ifndef POISSON_PYTHONIFY_H
#include <Interface/Interface.h>
#include "memory/python_tensor.h"
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;
namespace py = nb;
using pyarray = nb::array<nb::c_contig>;
using pyarray_c = nb::array<libpoisson::real, nb::c_contig>;
namespace lp = libpoisson::python;
using libpoisson::real;
using Parameters = libpoisson::Parameters;
using Configuration = libpoisson::Configuration;

#define POISSONSTR(s) xPOISSONSTR(s)
#define xPOISSONSTR(s) #s


