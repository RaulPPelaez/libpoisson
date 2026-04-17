#include "Interactor/SpectralEwaldPoisson.cuh"
#include "Interactor/DoublyPeriodic/DPPoissonSlab.cuh"
#include "uammd.cuh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <thrust/transform.h>
#include <memory>
using real = float;

namespace uammd {
    class ParticleData;
    class Interactor;
} // namespace uammd
class PSEWrapper {
    std::shared_ptr<uammd::ParticleData> pd;
    std::shared_ptr<uammd::Poisson> solver;
    real Lx;
    real Ly;
    real Lz;
    real perm;
    real gw;
    real tolerance;
    real split;

    public:
    PSEWrapper(real Lx, real Ly, real Lz, real perm, real gw, real tolerance = 1e-3, real split = -1.0)
        : Lx(Lx), Ly(Ly), Lz(Lz), perm(perm), gw(gw), tolerance(tolerance), split(split) {}
    void compute_poisson(real *pos, real *charge, real *field, real *potential,
            int numberParticles);
};
class DPSlabWrapper{
    std::shared_ptr<uammd::ParticleData> pd;
    std::shared_ptr<uammd::DPPoissonSlab> solver;
    real Lx;
    real Ly;
    real Lz;
    real permInside;
    real permTop;
    real permBottom;
    real gw;
    real tolerance;
    real split;
    real totalCharge;
    public:
    DPSlabWrapper(real Lx, real Ly, real Lz, real permInside, real permTop, real permBottom, real gw, real tolerance = 1e-3, real split = -1.0)
        : Lx(Lx), Ly(Ly), Lz(Lz), permInside(permInside), permTop(permTop), permBottom(permBottom), gw(gw), tolerance(tolerance), split(split) {}
    void compute_poisson(real *pos, real *charge, real *field, real *potential,
            int numberParticles);
};

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

auto createPSEInteractor(std::shared_ptr<ParticleData> pd, real Lx, real Ly, real Lz,
        real permittivity, real gaussianWidth, real tolerance, real split = -1) {
    Poisson::Parameters par;
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
    auto solver = std::make_shared<Poisson>(pd, par);
    return solver;
}

auto createDPSlabInteractor(std::shared_ptr<ParticleData> pd, real Lx, real Ly, real Lz,
        real permittivityInside, real permittivityTop, real permittivityBottom,
        real gaussianWidth, real tolerance, real split = -1){
    DPPoissonSlab::Parameters par;
    par.Lxy = make_real2(Lx, Ly);
    par.H = Lz;
    DPPoissonSlab::Permitivity perm;
    perm.inside = permittivityInside;
    perm.top = permittivityTop;
    perm.bottom = permittivityBottom;
    par.permitivity = perm;
    par.gw = gaussianWidth;
    par.split = split;
    auto solver = std::make_shared<DPPoissonSlab>(pd, par);
    return solver;
}

void PSEWrapper::compute_poisson(real *i_pos, real *i_charge, real *i_field, real *i_potential,
        int numberParticles) {
    if (!pd || numberParticles != pd->getNumParticles()) {
        pd = std::make_shared<ParticleData>(numberParticles);
        solver = createPSEInteractor(pd, Lx, Ly, Lz, perm, gw, tolerance, split);
    }

    {
        auto pos = pd->getPos(access::gpu, access::write);
        const real3 *i_pos3 = reinterpret_cast<real3 *>(i_pos);
        thrust::transform(thrust::cuda::par, i_pos3, i_pos3 + numberParticles,
                pos.begin(), ToReal4());
        auto charge = pd->getCharge(access::gpu, access::write);
        thrust::copy(thrust::cuda::par, i_charge, i_charge + numberParticles,
                charge.begin());
    }
    auto fieldPotential = solver->computeFieldPotentialAtParticles();
    //This assumes that the field and potential are stored in a contiguous array, with the field being 3 components per particle and the potential being 1 component per particle. If this is not the case, this code will need to be modified to account for the actual layout of the data.
    thrust::for_each(thrust::cuda::par,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(numberParticles),
            [fp = thrust::raw_pointer_cast(fieldPotential.data()),
            field = i_field,
            pot = i_potential] __device__ (int i) {

            real4 v = fp[i];

            field[3*i + 0] = v.x;
            field[3*i + 1] = v.y;
            field[3*i + 2] = v.z;

            pot[i] = v.w;
            });
}

void DPSlabWrapper::compute_poisson(real *i_pos, real *i_charge, real *i_field, real *i_potential,
        int numberParticles) {
    if (!pd || numberParticles != pd->getNumParticles()) { //Only first time compute_poisson is called, we create the ParticleData and the solver. This is because the solver needs to know the number of particles to create the necessary data structures, and we don't want to recreate these data structures every time compute_poisson is called, as that would be very inefficient.
        pd = std::make_shared<ParticleData>(numberParticles);
        totalCharge = thrust::reduce(thrust::cuda::par, i_charge, i_charge + numberParticles, 0.0f, thrust::plus<real>());
        {
            auto charge = pd->getCharge(access::gpu, access::write);
            thrust::copy(thrust::cuda::par, i_charge, i_charge + numberParticles,
                    charge.begin());
        }
        solver = createDPSlabInteractor(pd, Lx, Ly, Lz, permInside, permTop, permBottom, gw, tolerance, split);
    }

    {
        auto pos = pd->getPos(access::gpu, access::write);
        const real3 *i_pos3 = reinterpret_cast<real3 *>(i_pos);
        thrust::transform(thrust::cuda::par, i_pos3, i_pos3 + numberParticles,
                pos.begin(), ToReal4());
        real totalCharge_given = thrust::reduce(thrust::cuda::par, i_charge, i_charge + numberParticles, 0.0f, thrust::plus<real>());
        if (abs(totalCharge_given - totalCharge) > tolerance) {
            throw nb::value_error(
                    "Total charge has changed since initialization. \n"
                    "Double Periodic systems requires a constant total charge due to performance optimizations. \n"
                    "If you need to change the total charge, you will need to create a new instance of the solver."
                    );
        }
        auto charge = pd->getCharge(access::gpu, access::write);
        thrust::copy(thrust::cuda::par, i_charge, i_charge + numberParticles,
                charge.begin());
    }
    auto fieldPotential = solver->computeFieldPotentialAtParticles();
    //cudaDeviceSynchronize();
    //This assumes that the field and potential are stored in a contiguous array, with the field being 3 components per particle and the potential being 1 component per particle. If this is not the case, this code will need to be modified to account for the actual layout of the data.
    thrust::for_each(thrust::cuda::par,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(numberParticles),
            [fp = thrust::raw_pointer_cast(fieldPotential.data()),
            field = i_field,
            pot = i_potential] __device__ (int i) {

            real4 v = fp[i];

            field[3*i + 0] = v.x;
            field[3*i + 1] = v.y;
            field[3*i + 2] = v.z;

            pot[i] = v.w;
            });
}


template <typename T>
using pyarray3 =
nb::ndarray<T, nb::cupy, nb::device::cuda, nb::c_contig, nb::shape<-1, 3>>;

template <typename T>
using pyarray = nb::ndarray<T, nb::cupy, nb::device::cuda, nb::c_contig>;

class PyPSE {
    std::shared_ptr<PSEWrapper> uammdPSE_wrapper;

    public:
    PyPSE(real Lx, real Ly, real Lz, real perm, real gw, real tolerance = 1e-3, real split = -1.0) {
        uammdPSE_wrapper = std::make_shared<PSEWrapper>(Lx, Ly, Lz, perm, gw, tolerance, split);
    }
    void compute_poisson(pyarray3<real> positions, pyarray<real> charges,
            pyarray3<real> field, pyarray<real> potential) {
        uammdPSE_wrapper->compute_poisson(positions.data(), charges.data(),
                field.data(), potential.data(), positions.shape(0));
    }
};

class PyDPSlab {
    std::shared_ptr<DPSlabWrapper> uammdDPSlab_wrapper;

    public:
    PyDPSlab(real Lx, real Ly, real Lz, real permInside, real permTop, real permBottom, real gw, real tolerance = 1e-3, real split = -1.0) {
        uammdDPSlab_wrapper = std::make_shared<DPSlabWrapper>(Lx, Ly, Lz, permInside, permTop, permBottom, gw, tolerance, split);
    }
    void compute_poisson(pyarray3<real> positions, pyarray<real> charges,
            pyarray3<real> field, pyarray<real> potential) {
        uammdDPSlab_wrapper->compute_poisson(positions.data(), charges.data(),
                field.data(), potential.data(), positions.shape(0));
    }
};

using namespace nb::literals;
NB_MODULE(_uammd, m) {
    m.doc() = "";
    auto uammd_pse = nb::class_<PyPSE>(m, "uammd_pse");
    uammd_pse.def(nb::init<real, real, real, real, real, real ,real>(), "Lx"_a, "Ly"_a, "Lz"_a, "permittivity"_a,
            "gaussian_width"_a, "tolerance"_a = 1e-3, "split"_a = -1.0)
        .def("compute_poisson", &PyPSE::compute_poisson, "positions"_a,
                "charges"_a, "fields"_a, "potentials"_a);
    auto uammd_dpslab = nb::class_<PyDPSlab>(m, "uammd_dpslab");
    uammd_dpslab.def(nb::init<real, real, real, real, real, real, real, real ,real>(), "Lx"_a, "Ly"_a, "Lz"_a,
            "permittivity_inside"_a, "permittivity_top"_a, "permittivity_bottom"_a,
            "gaussian_width"_a, "tolerance"_a = 1e-3, "split"_a = -1.0)
        .def("compute_poisson", &PyDPSlab::compute_poisson, "positions"_a,
                "charges"_a, "fields"_a, "potentials"_a);
}
