Default Solvers
---------------

.. list-table::
   :header-rows: 1

   * - Boundary Conditions
     - Device
     - Solver
     - Extra Parameters
   * - .. code-block:: python

		("open",

		 "open",

		 "open")
     - .. code-block:: python

		"cuda"
     - :class:`NBody <libPoisson.solvers_cuda.nbody.NBody>`
     - 
   * - .. code-block:: python

		("open",

		 "open",

		 "single_wall")
     - .. code-block:: python

		"cuda"
     - :class:`NBodySingleWall <libPoisson.solvers_cuda.nbody.NBodySingleWall>`
     - .. code-block:: python

		bottom_wall_position: float,
		bottom_permittivity: float
   * - .. code-block:: python

		("open",

		 "open",

		 "double_wall")
     - .. code-block:: python

		"cuda"
     - :class:`NBodyDoubleWall <libPoisson.solvers_cuda.nbody.NBodyDoubleWall>`
     - .. code-block:: python

		bottom_wall_position: float,
		bottom_permittivity: float,
		top_wall_position: float,
		top_permittivity: float,
		tolerance: float = 1e-05
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "open")
     - .. code-block:: python

		"cuda"
     - :class:`UAMMDPoissonSlabOpen <libPoisson.solvers_cuda.uammd_dpslab.UAMMDPoissonSlabOpen>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "single_wall")
     - .. code-block:: python

		"cuda"
     - :class:`UAMMDPoissonSlabSingleWall <libPoisson.solvers_cuda.uammd_dpslab.UAMMDPoissonSlabSingleWall>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06,
		bottom_permittivity: float
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "double_wall")
     - .. code-block:: python

		"cuda"
     - :class:`UAMMDPoissonSlabDoubleWall <libPoisson.solvers_cuda.uammd_dpslab.UAMMDPoissonSlabDoubleWall>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06,
		bottom_permittivity: float,
		top_permittivity: float
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "periodic")
     - .. code-block:: python

		"cuda"
     - :class:`UAMMDSplitEwaldPoisson <libPoisson.solvers_cuda.uammd_pse.UAMMDSplitEwaldPoisson>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06

All Solvers
-----------

.. list-table::
   :header-rows: 1

   * - Boundary Conditions
     - Device
     - Implementation
     - Solver
     - Extra Parameters
   * - .. code-block:: python

		("open",

		 "open",

		 "open")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"numba"
     - :class:`NBody <libPoisson.solvers_cuda.nbody.NBody>`
     - 
   * - .. code-block:: python

		("open",

		 "open",

		 "single_wall")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"numba"
     - :class:`NBodySingleWall <libPoisson.solvers_cuda.nbody.NBodySingleWall>`
     - .. code-block:: python

		bottom_wall_position: float,
		bottom_permittivity: float
   * - .. code-block:: python

		("open",

		 "open",

		 "double_wall")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"numba"
     - :class:`NBodyDoubleWall <libPoisson.solvers_cuda.nbody.NBodyDoubleWall>`
     - .. code-block:: python

		bottom_wall_position: float,
		bottom_permittivity: float,
		top_wall_position: float,
		top_permittivity: float,
		tolerance: float = 1e-05
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "open")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"uammd"
     - :class:`UAMMDPoissonSlabOpen <libPoisson.solvers_cuda.uammd_dpslab.UAMMDPoissonSlabOpen>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "single_wall")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"uammd"
     - :class:`UAMMDPoissonSlabSingleWall <libPoisson.solvers_cuda.uammd_dpslab.UAMMDPoissonSlabSingleWall>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06,
		bottom_permittivity: float
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "double_wall")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"uammd"
     - :class:`UAMMDPoissonSlabDoubleWall <libPoisson.solvers_cuda.uammd_dpslab.UAMMDPoissonSlabDoubleWall>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06,
		bottom_permittivity: float,
		top_permittivity: float
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "periodic")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"uammd"
     - :class:`UAMMDSplitEwaldPoisson <libPoisson.solvers_cuda.uammd_pse.UAMMDSplitEwaldPoisson>`
     - .. code-block:: python

		L: Sequence,
		splitting_ratio: float = -1.0,
		tolerance: float = 1e-06
   * - .. code-block:: python

		("periodic",

		 "periodic",

		 "periodic")
     - .. code-block:: python

		"cuda"
     - .. code-block:: python

		"spreadinterp"
     - :class:`SpreadInterp <libPoisson.solvers_cuda.spreadinterp.SpreadInterp>`
     - .. code-block:: python

		gaussian_cutoff: float,
		L: Sequence,
		n_grid: Sequence