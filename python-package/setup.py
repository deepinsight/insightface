#!/usr/bin/env python
from Cython.Build import cythonize
from setuptools import Extension, setup
import numpy as np

extensions = [
    Extension(
        "insightface.thirdparty.face3d.mesh.cython.mesh_core_cython",
        [
            "insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.pyx",
            "insightface/thirdparty/face3d/mesh/cython/mesh_core.cpp",
        ],
        language="c++",
    ),
]
ext_modules = cythonize(extensions)

setup(ext_modules=ext_modules, include_dirs=[np.get_include()])
