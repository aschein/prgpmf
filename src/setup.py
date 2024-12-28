import os
import sys

# from setuptools import setup, Extension
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
from pathlib import Path  # Use pathlib instead of path module

# Include and library paths for GSL
include_gsl_dir = '/usr/include/gsl'
lib_gsl_dir = '/usr/lib/aarch64-linux-gnu'

def make_extension(ext_name, ext_path=None):
    """
    Creates a Cython Extension.
    """
    if ext_path is None:
        ext_path = Path(*ext_name.split('.'))  # Fix path construction
        ext_path = str(ext_path) + '.pyx'
    assert Path(ext_path).is_file()

    return Extension(
        name=ext_name,
        sources=[str(ext_path)],  # Ensure paths are strings
        include_dirs=[np.get_include(), include_gsl_dir],
        library_dirs=[lib_gsl_dir],
        libraries=['gsl', 'gslcblas'],
        extra_compile_args=['-fopenmp', '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
        extra_link_args=['-fopenmp']
    )

def get_pkgs_and_exts(dir="."):
    """
    Finds packages and extensions for Cython compilation.
    """
    pkgs = set()
    exts = []

    for ext_path in Path(dir).rglob("*.pyx"):
        # Get the relative path and construct the module name
        relative_path = ext_path.relative_to(Path(dir))
        subdirs = relative_path.parts[:-1]  # Exclude the file name
        ext_name = relative_path.stem  # Get the base name without extension

        # Construct package and module names
        pkg = ".".join(subdirs)  # Package name
        ext_name = f"{pkg}.{ext_name}" if pkg else ext_name  # Full module name

        if pkg:
            pkgs.add(pkg)
        exts.append(make_extension(ext_name, ext_path))

    return list(pkgs), exts

# Collect packages and extensions
pkgs, exts = get_pkgs_and_exts(dir=".")

# Setup configuration
setup(
    name='prgpmf',
    version='1.0.0',
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(exts, language_level=3, compiler_directives={"boundscheck": False}),
    packages=pkgs,
    package_dir={"": "."},  # Adjust this to the correct root directory
    install_requires=["click"],
    entry_points={
        "console_scripts": [
            "prgpmf = prgpmf.cli:main",
        ]
    },
    description="Poisson-randomized gamma Poisson matrix factorization.",
    author="Aaron Schein",
    zip_safe=False
)



