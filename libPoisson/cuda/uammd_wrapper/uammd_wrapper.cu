#include "Interactor/Interactor.cuh"
#include "CustomPSE.cuh"
//#include "Interactor/SpectralEwaldPoisson.cuh"
#include "common.h"
#include "uammd.cuh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <thrust/transform.h>
using namespace uammd;
namespace nb = nanobind;

struct ToReal3 {
    template <class T> __device__ real3 operator()(T v) { return make_real3(v); }
};

struct ToReal4 {
    template <class T> __device__ auto operator()(T v) { return make_real4(v); }
};

struct SumReal3ToReal4 {
    __device__ real4 operator()(real3 in, real4 other) {
        return make_real4(in.x + other.x, in.y + other.y, in.z + other.z, other.w);
    }
};

auto createElectrostaticInteractor(std::shared_ptr<ParticleData> pd, real Lx, real Ly, real Lz,
        real permittivity, real gaussianWidth, real tolerance = 1e-3, real split = -1) {
    using Electro = CustomPoisson;
    Electro::Parameters par;
    par.box = Box(make_real3(Lx, Ly, Lz));
    par.epsilon = permittivity;
    par.tolerance = tolerance;
    par.gw = gaussianWidth;
    par.split = split;
    real lbox = std::min({Lx, Ly, Lz});
    int maxcells = lbox / par.gw;
    if (maxcells >= 128 && par.split < 0) {
        // So the user doesn't have to worry about the split parameter, we set it to a reasonable value based on the box size and the number of cells. This is not optimal, but it should work well for most cases.
        par.split = 0.3 / par.gw;
    }
    auto elec = std::make_shared<Electro>(pd, par);
    return elec;
}

void UAMMDWrapper::compute_poisson(real *i_pos, real *i_charge, real *i_force, real *i_energy,
        int numberParticles) {
    if (!pd || numberParticles != pd->getNumParticles()) {
        pd = std::make_shared<ParticleData>(numberParticles);
        poisson = createElectrostaticInteractor(pd, Lx, Ly, Lz, perm, gw, tolerance, split);
    }

    {
        auto pos = pd->getPos(access::gpu, access::write);
        const real3 *i_pos3 = reinterpret_cast<real3 *>(i_pos);
        thrust::transform(thrust::cuda::par, i_pos3, i_pos3 + numberParticles,
                pos.begin(), ToReal4());
        auto charge = pd->getCharge(access::gpu, access::write);
        thrust::copy(thrust::cuda::par, i_charge, i_charge + numberParticles,
                charge.begin());
        auto forces = pd->getForce(access::gpu, access::write);
        thrust::fill(thrust::cuda::par, forces.begin(), forces.end(), real4{});

        auto energy = pd->getEnergy(access::gpu, access::write);
        thrust::fill(thrust::cuda::par, energy.begin(), energy.end(), 0.0);
    }
    poisson->sum({.force = true, .energy = true}, 0);

    {
        auto forces = pd->getForce(access::gpu, access::read);
        real3 *i_force3 = reinterpret_cast<real3 *>(i_force);
        thrust::transform(thrust::cuda::par, forces.begin(), forces.end(), i_force3,
                ToReal3());
        auto energy = pd->getEnergy(access::gpu, access::read);
        thrust::copy(thrust::cuda::par, energy.begin(), energy.end(), i_energy);
    }
}
template <typename T>
using pyarray3 =
nb::ndarray<T, nb::cupy, nb::device::cuda, nb::c_contig, nb::shape<-1, 3>>;

template <typename T>
using pyarray = nb::ndarray<T, nb::cupy, nb::device::cuda, nb::c_contig>;

class PyPoisson {
    std::shared_ptr<UAMMDWrapper> uammd_wrapper;

    public:
    PyPoisson(real Lx, real Ly, real Lz, real perm, real gw, real tolerance = 1e-3, real split = -1.0) {
        uammd_wrapper = std::make_shared<UAMMDWrapper>(Lx, Ly, Lz, perm, gw, tolerance, split);
    }
    void compute_poisson(pyarray3<real> positions, pyarray<real> charges,
            pyarray3<real> forces, pyarray<real> energy) {
        uammd_wrapper->compute_poisson(positions.data(), charges.data(),
                forces.data(), energy.data(), positions.shape(0));
    }
};

using namespace nb::literals;
NB_MODULE(_uammd, m) {
    m.doc() = "";
    auto solver = nb::class_<PyPoisson>(m, "uammd_pse");
    solver.def(nb::init<real, real, real, real, real, real ,real>(), "Lx"_a, "Ly"_a, "Lz"_a, "permittivity"_a,
            "gaussian_width"_a, "tolerance"_a = 1e-3, "split"_a = -1.0)
        .def("compute_poisson", &PyPoisson::compute_poisson, "positions"_a,
                "charges"_a, "forces"_a, "energy"_a);
}
