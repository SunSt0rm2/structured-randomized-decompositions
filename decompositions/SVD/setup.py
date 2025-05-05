### Run in terminal with transforms.cpp file in same folder: python setup.py build_ext --inplace

import sys
import os
from setuptools import setup, Extension
from sysconfig import get_paths

# Get include paths
paths = get_paths()
python_include = paths['include']

# Try to get pybind11 include path
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = os.path.join(sys.prefix, 'include')

# Additional include directories
include_dirs = [
    python_include,
    pybind11_include,
    os.path.join(sys.prefix, 'include')
]

# Filter out non-existent directories
include_dirs = [d for d in include_dirs if os.path.exists(d)]

# Extension module
ext_module = Extension(
    'transforms',
    sources=['transforms.cpp'],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=['-std=c++11', '-O3'],
)

setup(
    name='transforms',
    version='0.1',
    ext_modules=[ext_module],
)