Installation
============

Compilation from source
-----------------------

For now, only available via compiling from source.

.. code-block:: bash

   git clone https://github.com/RaulPPelaez/libPoisson.git

The file `environment.yml` contains the dependencies for this project. You can create a new conda environment with these dependencies using the following command:

.. code-block:: bash

   conda env create -f environment.yml

Then activate the environment:

.. code-block:: bash

   conda activate libpoisson

Then install the library with pip:

.. code-block:: bash

   pip install .
