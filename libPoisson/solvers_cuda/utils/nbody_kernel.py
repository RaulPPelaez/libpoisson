import numba
from numba import cuda, float32
import numpy as np
import math

# ----------------------------------
# Device function: Coulomb
# ----------------------------------
@cuda.jit(device=True)
def coulomb_interaction(tx, ty, tz, sx, sy, sz, sq, a):
    dx = sx - tx
    dy = sy - ty
    dz = sz - tz

    r2 = dx*dx + dy*dy + dz*dz + 1e-12
    r = math.sqrt(r2)

    r_sigma = r / a
    erf_term = math.erf(r_sigma)

    inv_r = 1.0 / r
    inv_r3 = inv_r * inv_r * inv_r

    coeff = inv_r3 * (erf_term - 2.0/math.sqrt(math.pi) * r_sigma * math.exp(-r_sigma*r_sigma))

    Ex = -sq * dx * coeff
    Ey = -sq * dy * coeff
    Ez = -sq * dz * coeff

    phi = sq * erf_term * inv_r

    return Ex, Ey, Ez, phi


# ----------------------------------
# Kernel
# ----------------------------------
@cuda.jit
def nbody_kernel(target, source, nt, ns, field_potential, a):
    i = cuda.grid(1)
    if i >= nt:
        return
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    phi = 0.0
    tx = target[i, 0]
    ty = target[i, 1]
    tz = target[i, 2]
    for j in range(ns):
        sx = source[j, 0]
        sy = source[j, 1]
        sz = source[j, 2]
        sq = source[j, 3]
        dEx, dEy, dEz, dphi = coulomb_interaction(tx, ty, tz, sx, sy, sz, sq, a)
        Ex += dEx
        Ey += dEy
        Ez += dEz
        phi += dphi
    field_potential[i, 0] += Ex
    field_potential[i, 1] += Ey
    field_potential[i, 2] += Ez
    field_potential[i, 3] += phi
