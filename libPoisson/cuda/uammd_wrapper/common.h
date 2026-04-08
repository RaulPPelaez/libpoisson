#pragma once
#include <memory>
using real = float;

namespace uammd {
    class ParticleData;
    class Interactor;
} // namespace uammd
class UAMMDWrapper {
    std::shared_ptr<uammd::ParticleData> pd;
    std::shared_ptr<uammd::Interactor> poisson;
    real Lx;
    real Ly;
    real Lz;
    real perm;
    real gw;
    real tolerance;
    real split;

    public:
    UAMMDWrapper(real Lx, real Ly, real Lz, real perm, real gw, real tolerance = 1e-3, real split = -1.0)
        : Lx(Lx), Ly(Ly), Lz(Lz), perm(perm), gw(gw), tolerance(tolerance), split(split) {}
    void compute_poisson(real *pos, real *charge, real *force, real *energy,
            int numberParticles);
};
