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
  real lbox;
  real perm;
  real gw;

public:
  UAMMDWrapper(real lbox, real perm, real gw)
      : lbox(lbox), perm(perm), gw(gw) {}
  void compute_poisson(real *pos, real *charge, real *force,
                       int numberParticles);
};
