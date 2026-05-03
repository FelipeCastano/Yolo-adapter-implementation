import os
import sys

# Añadir paths de los módulos
sys.path.insert(0, os.path.abspath('../../src/model'))
sys.path.insert(0, os.path.abspath('../../src/data_management'))

project   = 'Electronic Symbol Detector'
copyright = '2026'
author    = 'Felipe'
release   = '1.0'

extensions = [
    'sphinx.ext.autodoc',      # genera docs desde docstrings
    'sphinx.ext.napoleon',     # soporte para formato :param: :type:
    'sphinx.ext.viewcode',     # links al código fuente
    'sphinx.ext.autosummary',  # tablas de resumen
]

html_theme = 'sphinx_rtd_theme'

autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'private-members':  False,
    'show-inheritance': True,
}

