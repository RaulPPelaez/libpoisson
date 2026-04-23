Quickstart
==========

Basic usage:

.. code-block:: python

    import libPoisson as lp
    import cupy as cp

    solver = lp.get_solver(boundary_conditions=("open", "open", "open"),
                           device="cuda",
                           charge_radius=0.1, permittivity=1.0) #**kwargs for specific solvers

    N = 100
    M = 100
    target_pos = cp.zeros(N, 3) # N target positions
    source_pos = cp.zeros(M, 3) # M source positions
    source_charge = cp.ones(M) # M source charges

    potential, field = solver.solve(source_pos, target_pos, source_charge)

The ``potential`` and ``field`` are the computed potential and electric field at the target positions, respectively. The solver can be configured with different boundary conditions, device options, and specific parameters for various solvers.

