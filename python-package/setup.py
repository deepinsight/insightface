#!/usr/bin/env python
import os
import io
import glob
import numpy
import re
import shutil
import sys
import subprocess
import platform
import logging
from setuptools import setup, find_packages
from distutils.core import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

pypandoc_enabled = True
try:
    import pypandoc
    print('pandoc enabled')
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError, ModuleNotFoundError):
    print('WARNING: pandoc not enabled')
    long_description = open('README.md').read()
    pypandoc_enabled = False

#import pypandoc
#long_description = pypandoc.convert('README.md', 'rst')
VERSION = find_version('insightface', '__init__.py')

requirements = [
    'numpy',
    'onnx',
    'tqdm',
    'requests',
    'matplotlib',
    'Pillow',
    'scipy',
    #'opencv-python',
    'scikit-learn',
    'scikit-image',
    'easydict',
    'cython',
    'albumentations',
    'prettytable',
]

extensions = [
        Extension("insightface.thirdparty.face3d.mesh.cython.mesh_core_cython", 
            ["insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.pyx", "insightface/thirdparty/face3d/mesh/cython/mesh_core.cpp"], language='c++'),
        ]
data_images = list(glob.glob('insightface/data/images/*.jpg'))
data_images += list(glob.glob('insightface/data/images/*.png'))

data_mesh = list(glob.glob('insightface/thirdparty/face3d/mesh/cython/*.h'))
data_mesh += list(glob.glob('insightface/thirdparty/face3d/mesh/cython/*.c'))
data_mesh += list(glob.glob('insightface/thirdparty/face3d/mesh/cython/*.py*'))

data_objects = list(glob.glob('insightface/data/objects/*.pkl'))

data_files = [ ('insightface/data/images', data_images) ]
data_files += [ ('insightface/data/objects', data_objects) ]
data_files += [ ('insightface/thirdparty/face3d/mesh/cython', data_mesh) ]

ext_modules=cythonize(extensions)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Check if running on macOS (Darwin)
if platform.system() == "Darwin":
    logging.info("Detected macOS. Checking if Homebrew, LLVM, and OpenMP are installed...")

    # Check if Homebrew is installed
    brew_check = subprocess.run(["which", "brew"], capture_output=True, text=True)
    if brew_check.returncode != 0:
        logging.warning("Homebrew is not installed. You may need to manually install dependencies.")
        logging.warning("   Install Homebrew: https://brew.sh/")
        logging.warning("   Then, run: brew install llvm libomp")
        logging.info("Proceeding without setting the compiler.")
    
    else:
        # Check if LLVM is installed
        llvm_check = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True)
        if llvm_check.returncode != 0:
            logging.warning("LLVM is not installed. This may cause installation issues.")
            logging.warning("   To install, run: brew install llvm libomp")
            logging.warning("   Then, restart the installation process.")
            logging.info("Proceeding without setting the compiler.")

        else:
            # Set compiler dynamically if LLVM is installed
            llvm_path = subprocess.getoutput("brew --prefix llvm")
            os.environ["CC"] = f"{llvm_path}/bin/clang"
            os.environ["CXX"] = f"{llvm_path}/bin/clang++"
            logging.info(f"Using compiler: {os.environ['CC']}")

setup(
    # Metadata
    name='insightface',
    version=VERSION,
    author='InsightFace Contributors',
    author_email='contact@insightface.ai',
    url='https://github.com/deepinsight/insightface',
    description='InsightFace Python Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    # Package info
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    data_files=data_files,
    zip_safe=True,
    include_package_data=True,
    entry_points={"console_scripts": ["insightface-cli=insightface.commands.insightface_cli:main"]},
    install_requires=requirements,
    headers=['insightface/thirdparty/face3d/mesh/cython/mesh_core.h'],
    ext_modules=ext_modules,
    include_dirs=numpy.get_include(),
)

print('pypandoc enabled:', pypandoc_enabled)

