#!/usr/bin/env python
import os
import io
import re
import sys
import subprocess
import platform
import logging
from setuptools import setup, find_namespace_packages


FACE3D_BUILD_FLAG = '--with-face3d'


def strtobool_env(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


build_face3d = strtobool_env(os.environ.get('INSIGHTFACE_WITH_FACE3D', ''))
if FACE3D_BUILD_FLAG in sys.argv:
    build_face3d = True
    sys.argv.remove(FACE3D_BUILD_FLAG)

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

long_description = read('README.md')
VERSION = find_version('insightface', '__init__.py')

requirements = [
    'numpy',
    'onnx>=1.13',
    'onnxruntime',
    'opencv-python',
    'tqdm',
    'requests',
    'scipy',
    #'opencv-python',
    'scikit-image',
]

privateframe_requirements = [
    'av>=12',
    'PyYAML>=6.0',
]

gui_only_requirements = [
    'PySide6-Essentials>=6.5',
    'Pillow>=9.1',
    'reportlab',
    'scikit-learn',
    'cryptography>=42.0.0',
    'rfc8785>=0.1.4',
]

gui_requirements = gui_only_requirements + privateframe_requirements

face3d_requirements = [
    'cython',
    'albumentations',
    'matplotlib',
]

package_data = {
    "insightface.data.images": ["*.jpg", "*.jpeg", "*.png"],
    "insightface.data.objects": ["*.pkl"],
    "insightface.gui.assets": ["*.png", "*.ico", "*.icns"],
    "insightface.app.privateframe": ["configs/*.yaml", "docs/*.md"],
    "insightface.model_zoo": ["trusted_keys/*.pem"],
}

packages = find_namespace_packages(
    include=(
        "insightface",
        "insightface.*",
        "insightface_privateframe_bootstrap",
    ),
    exclude=("docs", "docs.*", "tests", "tests.*", "scripts", "scripts.*"),
)
if not build_face3d:
    packages = [
        package
        for package in packages
        if package != "insightface.thirdparty.face3d.mesh.cython"
    ]

ext_modules = []
include_dirs = []
headers = []
if build_face3d:
    import numpy
    from distutils.core import Extension
    from Cython.Build import cythonize

    extensions = [
            Extension("insightface.thirdparty.face3d.mesh.cython.mesh_core_cython",
                ["insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.pyx", "insightface/thirdparty/face3d/mesh/cython/mesh_core.cpp"], language='c++'),
            ]
    package_data["insightface.thirdparty.face3d.mesh.cython"] = [
        "*.h",
        "*.c",
        "*.cpp",
        "*.pyx",
        "*.py",
    ]
    ext_modules = cythonize(extensions)
    include_dirs = [numpy.get_include()]
    headers = ['insightface/thirdparty/face3d/mesh/cython/mesh_core.h']

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Check if running on macOS (Darwin)
if build_face3d and platform.system() == "Darwin":
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
            llvm_cc = f"{llvm_path}/bin/clang"
            llvm_cxx = f"{llvm_path}/bin/clang++"
            if os.path.exists(llvm_cc) and os.path.exists(llvm_cxx):
                os.environ["CC"] = llvm_cc
                os.environ["CXX"] = llvm_cxx
                logging.info(f"Using compiler: {os.environ['CC']}")
            else:
                logging.warning("Homebrew LLVM path exists but clang/clang++ was not found.")
                logging.info("Falling back to the default system compiler.")

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
    # Package info
    python_requires='>=3.10',
    packages=packages,
    package_data=package_data,
    zip_safe=True,
    include_package_data=False,
    entry_points={
        "console_scripts": [
            "insightface-cli=insightface.commands.insightface_cli:main",
            "insightface-gui=insightface.gui.__main__:main",
            "insightface-eval-studio=insightface.gui.__main__:main",
            "insightface-desktop=insightface.gui.__main__:main",
            "insightface-privateframe=insightface_privateframe_bootstrap:main",
        ]
    },
    extras_require={
        "gui": gui_requirements,
        "privateframe": privateframe_requirements,
        "face3d": face3d_requirements,
    },
    install_requires=requirements,
    headers=headers,
    ext_modules=ext_modules,
    include_dirs=include_dirs,
)

print('face3d build enabled:', build_face3d)
