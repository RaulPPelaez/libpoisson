project = 'libPoisson'
copyright = '2026, Pablo Diez-Silva'
author = 'Pablo Diez-Silva'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import sys
import os
path = os.path.abspath('../../libPoisson')
sys.path.insert(0, path)
print(f"--- SPHINX ESTÁ BUSCANDO EN: {path} ---")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from auto_factory.tables import build_solver_tables
from auto_factory.solvers import build_solver_classes
build_solver_classes()
build_solver_tables()
