import libpoisson as lp
import cupy as cp
import numpy as np

number_particles = 1000
lbox = 10.0
positions = cp.random.random((number_particles, 3)) * lbox
charges = cp.random.choice([-1, 1], size=number_particles)
gaussian_width = 0.1

s = lp.PoissonSolver(lbox=lbox, permittivity=1.0, gaussian_width=gaussian_width)

forces = s.compute_poisson(positions=positions, charges=charges)

print(f"Forces computed for {number_particles} particles in a box of size {lbox}:")
print(forces)
