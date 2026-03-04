def test_import():
    import libPoisson as lp

import libPoisson as lp
import numpy as np
def test_interface():
    solver = lp.NBody()
    solver.initialize(permitivity=1.0, chargeRadius=0.1)
    solver.setParameters()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    solver.setPositions(pos)
    charges = np.array([1.0, -1.0])
    phi = solver.Gdot(charges)

