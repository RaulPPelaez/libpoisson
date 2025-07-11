#include "common.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
namespace nb = nanobind;

template <typename T>
using pyarray3 =
    nb::ndarray<T, nb::cupy, nb::device::cuda, nb::c_contig, nb::shape<-1, 3>>;

template <typename T>
using pyarray = nb::ndarray<T, nb::cupy, nb::device::cuda, nb::c_contig>;

class PyPoisson {
  std::shared_ptr<UAMMDWrapper> uammd_wrapper;

public:
  PyPoisson(real lbox, real perm, real gw) {
    uammd_wrapper = std::make_shared<UAMMDWrapper>(lbox, perm, gw);
  }
  void compute_poisson(pyarray3<real> positions, pyarray<real> charges,
                       pyarray3<real> forces) {
    uammd_wrapper->compute_poisson(positions.data(), charges.data(),
                                   forces.data(), positions.shape(0));
  }
};

using namespace nb::literals;
NB_MODULE(_libpoisson, m) {
  m.doc() = "";
  auto solver = nb::class_<PyPoisson>(m, "PoissonSolver");
  solver.def(nb::init<real, real, real>(), "lbox"_a, "permittivity"_a,
             "gaussian_width"_a)
  .def("compute_poisson", &PyPoisson::compute_poisson, "positions"_a,
       "charges"_a, "forces"_a);
}
