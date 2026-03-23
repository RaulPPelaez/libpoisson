from numba import cuda, float64
from math import sqrt, erf, exp, pi

@cuda.jit
def nbody_kernel(source_pos, target_pos, charges, a, eps4pi, pot, field):
    i = cuda.grid(1)  # Cada hilo corresponde a un target particle
    M = target_pos.shape[0] // 3
    N = source_pos.shape[0] // 3

    if i >= M:
        return

    # Extraer posición del target
    x_i = target_pos[3*i]
    y_i = target_pos[3*i + 1]
    z_i = target_pos[3*i + 2]

    pot_val = 0.0
    fx, fy, fz = 0.0, 0.0, 0.0

    for j in range(N):
        x_j = source_pos[3*j]
        y_j = source_pos[3*j + 1]
        z_j = source_pos[3*j + 2]
        q_j = charges[j]

        dx = x_i - x_j
        dy = y_i - y_j
        dz = z_i - z_j
        r2 = dx*dx + dy*dy + dz*dz + 1e-20
        r = sqrt(r2)
        r_sigma = r / a
        erf_term = erf(r_sigma)

        # Potencial
        pot_val += q_j * erf_term / (eps4pi * r)

        # Campo
        inv_r3 = 1 / (r2 * r)
        coeff = q_j * inv_r3 / eps4pi * (erf_term - 2/sqrt(pi) * r_sigma * exp(-r_sigma*r_sigma))
        fx += coeff * dx
        fy += coeff * dy
        fz += coeff * dz

    pot[i] = pot_val
    field[3*i] = fx
    field[3*i+1] = fy
    field[3*i+2] = fz
