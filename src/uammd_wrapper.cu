#include "Interactor/Interactor.cuh"
#include "Interactor/SpectralEwaldPoisson.cuh"
#include "common.h"
#include "uammd.cuh"
#include <thrust/transform.h>
using namespace uammd;

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

auto createElectrostaticInteractor(std::shared_ptr<ParticleData> pd, real lbox,
                                   real permittivity, real gaussianWidth) {
  using Electro = Poisson;
  Electro::Parameters par;
  par.box = Box(lbox);
  par.epsilon = permittivity;
  par.tolerance = 1e-3;
  par.gw = gaussianWidth;
  int maxcells = lbox / par.gw;
  if (maxcells >= 128) {
    par.split = 0.3 / par.gw;
  }
  auto elec = std::make_shared<Electro>(pd, par);
  return elec;
}

void UAMMDWrapper::compute_poisson(real *i_pos, real *i_charge, real *i_force,
                                   int numberParticles) {
  if (!pd || numberParticles != pd->getNumParticles()) {
    pd = std::make_shared<ParticleData>(numberParticles);
    poisson = createElectrostaticInteractor(pd, lbox, perm, gw);
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
  }
  poisson->sum({.force = true}, 0);

  {
    auto forces = pd->getForce(access::gpu, access::read);
    real3 *i_force3 = reinterpret_cast<real3 *>(i_force);
    thrust::transform(thrust::cuda::par, forces.begin(), forces.end(), i_force3,
                      ToReal3());
  }
}
