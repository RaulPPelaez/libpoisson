import numba
from numba import cuda
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
def nbody_kernel(pos_target, pos_source, field_potential, a):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    N_target = pos_target.shape[0]
    N_source = pos_source.shape[0]

    if tid >= N_target:
        return

    tx = pos_target[tid, 0]
    ty = pos_target[tid, 1]
    tz = pos_target[tid, 2]

    Ex_total = 0.0
    Ey_total = 0.0
    Ez_total = 0.0
    phi_total = 0.0

    threads_per_block = cuda.blockDim.x

    # Shared memory dinámica (float32)
    sh_src = cuda.shared.array(shape=0, dtype=numba.float32)

    num_tiles = (N_source + threads_per_block - 1) // threads_per_block

    for tile in range(num_tiles):

        i_load = tile * threads_per_block + cuda.threadIdx.x

        # Carga en shared memory
        if i_load < N_source:
            base = cuda.threadIdx.x * 4
            sh_src[base + 0] = pos_source[i_load, 0]
            sh_src[base + 1] = pos_source[i_load, 1]
            sh_src[base + 2] = pos_source[i_load, 2]
            sh_src[base + 3] = pos_source[i_load, 3]

        cuda.syncthreads()

        tile_size = min(threads_per_block, N_source - tile * threads_per_block)

        for j in range(tile_size):
            base = j * 4

            sx = sh_src[base + 0]
            sy = sh_src[base + 1]
            sz = sh_src[base + 2]
            sq = sh_src[base + 3]

            Ex, Ey, Ez, phi = coulomb_interaction(tx, ty, tz, sx, sy, sz, sq, a)

            Ex_total += Ex
            Ey_total += Ey
            Ez_total += Ez
            phi_total += phi

        cuda.syncthreads()

    field_potential[tid, 0] = Ex_total
    field_potential[tid, 1] = Ey_total
    field_potential[tid, 2] = Ez_total
    field_potential[tid, 3] = phi_total
