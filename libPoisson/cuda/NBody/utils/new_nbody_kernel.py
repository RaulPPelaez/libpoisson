import numba
from numba import cuda
import numpy as np
from math import sqrt, erf, exp, pi
# ----------------------------------
# Device function: interaccion Coulomb
# ----------------------------------
@cuda.jit(device=True)
def coulomb_interaction(target, source, a):
    """
    target: (x,y,z)
    source: (x,y,z,q)
    Devuelve contribución de source a campo y potencial
    """
    dx = source[0] - target[0]
    dy = source[1] - target[1]
    dz = source[2] - target[2]
    r2 = dx*dx + dy*dy + dz*dz + 1e-12  # evitar division por cero
    r = sqrt(r2)
    r_sigma = r / a
    erf_term = erf(r_sigma)
    inv_r = 1.0 / r
    inv_r3 = inv_r * inv_r * inv_r
    q = source[3]

    # Campo eléctrico E = q * r / |r|^3
    coeff = inv_r3 * (erf_term - 2/sqrt(pi) * r_sigma * exp(-r_sigma*r_sigma))
    Ex = -q * dx * coeff
    Ey = -q * dy * coeff
    Ez = -q * dz * coeff

    # Potencial φ = q / |r|
    phi = q * erf_term * inv_r

    return Ex, Ey, Ez, phi

# ----------------------------------
# Kernel principal
# ----------------------------------
@cuda.jit
def nbody_kernel(pos_target, pos_source, field_potential, a):
    """
    pos_target: Nx3
    pos_source: Mx4 (x,y,z,q)
    field_potential: Nx4 output (Ex,Ey,Ez,phi)
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    N_target = pos_target.shape[0]
    N_source = pos_source.shape[0]

    if tid >= N_target:
        return

    # Target actual
    tx = pos_target[tid,0]
    ty = pos_target[tid,1]
    tz = pos_target[tid,2]

    Ex_total = 0.0
    Ey_total = 0.0
    Ez_total = 0.0
    phi_total = 0.0

    # Shared memory tile
    threads_per_block = cuda.blockDim.x
    sh_src = cuda.shared.array(0, dtype=numba.float64)  # dinámica

    num_tiles = (N_source + threads_per_block - 1) // threads_per_block

    for tile in range(num_tiles):
        # Cargar tile de sources a shared memory
        i_load = tile * threads_per_block + cuda.threadIdx.x
        if i_load < N_source:
            sh_src[cuda.threadIdx.x*4 + 0] = pos_source[i_load,0]
            sh_src[cuda.threadIdx.x*4 + 1] = pos_source[i_load,1]
            sh_src[cuda.threadIdx.x*4 + 2] = pos_source[i_load,2]
            sh_src[cuda.threadIdx.x*4 + 3] = pos_source[i_load,3]
        cuda.syncthreads()

        # Loop sobre tile
        #tile_size = min(threads_per_block, N_source - tile*threads_per_block)
        remaining = N_source - tile*threads_per_block
        tile_size = threads_per_block if remaining > threads_per_block else remaining
        for j in range(tile_size):
            sx = sh_src[j*4 + 0]
            sy = sh_src[j*4 + 1]
            sz = sh_src[j*4 + 2]
            sq = sh_src[j*4 + 3]
            Ex, Ey, Ez, phi = coulomb_interaction((tx,ty,tz), (sx,sy,sz,sq), a)
            Ex_total += Ex
            Ey_total += Ey
            Ez_total += Ez
            phi_total += phi
        cuda.syncthreads()

    # Guardar resultados
    field_potential[tid,0] = Ex_total
    field_potential[tid,1] = Ey_total
    field_potential[tid,2] = Ez_total
    field_potential[tid,3] = phi_total
