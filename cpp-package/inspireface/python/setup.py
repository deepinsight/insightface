import os
import platform
from pathlib import Path

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel


PYTHON_ROOT = Path(__file__).resolve().parent


def get_version() -> str:
    """Get version number"""
    version_path = PYTHON_ROOT / "version.txt"
    try:
        with version_path.open("r", encoding="utf-8") as file_handle:
            return file_handle.read().strip()
    except FileNotFoundError:
        return "0.0.0"


def get_post_version() -> str:
    """Get post version number"""
    post_path = PYTHON_ROOT / "post"
    try:
        with post_path.open("r", encoding="utf-8") as file_handle:
            return file_handle.read().strip()
    except FileNotFoundError:
        return ""


def get_wheel_platform_tag() -> str:
    """Get wheel package platform tag"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    arch_mapping = {
        'x86_64': {
            'windows': 'win_amd64',
            'linux': 'manylinux2014_x86_64',
            'darwin': 'macosx_12_0_x86_64'
        },
        'amd64': {
            'windows': 'win_amd64',
            'linux': 'manylinux2014_x86_64',
            'darwin': 'macosx_12_0_x86_64'
        },
        'arm64': {
            'windows': 'win_arm64',
            'linux': 'manylinux2014_aarch64',
            'darwin': 'macosx_11_0_arm64'
        },
        'aarch64': {
            'windows': 'win_arm64',
            'linux': 'manylinux2014_aarch64',
            'darwin': 'macosx_11_0_arm64'
        }
    }
    platform_override = os.getenv("INSPIRE_FACE_TARGET_AARCH_MAPPING")
    if platform_override:
        platform_arch = platform_override
    else:
        platform_arch = arch_mapping.get(machine, {}).get(system)
    if not platform_arch:
        raise RuntimeError("Unsupported platform: {} {}".format(system, machine))
    return platform_arch


def get_lib_path_info():
    """Get library file path information"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    supported_systems = {"windows", "linux", "darwin"}
    architecture_mapping = {
        "amd64": "x64",
        "x86_64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    arch = architecture_mapping.get(machine)
    if system not in supported_systems or arch is None:
        raise RuntimeError(f"Unsupported platform: system={system}, machine={machine}")
    return system, arch


class BinaryDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # Mark this is not a pure Python package
        self.root_is_pure = False
        # Set platform tag
        self.plat_name = get_wheel_platform_tag()
        self.plat_name_supplied = True
        self.universal = False

    def get_tag(self):
        # The wrapper uses ctypes and has no CPython ABI dependency. The wheel
        # is platform-specific because of the bundled native library, but one
        # build is reusable by every supported Python 3 version.
        return "py3", "none", self.plat_name


def get_target_platform_for_envs():
    """Get target platform for environments"""
    system = os.environ.get("INSPIRE_FACE_TARGET_PLATFORM")
    if system is None:
        system = get_lib_path_info()[0]
    
    machine = os.environ.get("INSPIRE_FACE_TARGET_ARCH")
    if machine is None:
        machine = get_lib_path_info()[1]
    return system, machine


# Get current platform information
system, arch = get_target_platform_for_envs()

# Build library file path relative to package
lib_path = os.path.join("modules", "core", "libs", system, arch, "*")

setup(
    name="inspireface",
    version=get_version() + get_post_version(),
    packages=find_packages(
        exclude=(
            "sample_testcase",
            "sample_testcase.*",
            "test",
            "test.*",
        )
    ),
    include_package_data=False,
    # package_data path should be relative to package directory
    package_data={
        "inspireface": [lib_path, "py.typed"]
    },
    install_requires=[
        "numpy",
        "loguru",
        "filelock",
        # Older ModelScope wheels omit Requires-Python: 1.22.1 uses Python
        # 3.8 syntax, and 1.29.2 unconditionally imports Python 3.9's zoneinfo.
        'modelscope<1.22.1; python_version < "3.8"',
        'modelscope<1.29.2; python_version == "3.8"',
        'modelscope; python_version >= "3.9"',
        'importlib-metadata; python_version < "3.8"',
    ],
    author="Jingyu Yan",
    author_email="tunmxy@163.com",
    description="InspireFace Python SDK",
    # Isolated wheel builds only contain this Python project, not its parent.
    long_description=(PYTHON_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/HyperInspire/InspireFace",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
    ],
    python_requires=">=3.7",
    cmdclass={
        "bdist_wheel": BinaryDistWheel,
    },
)
