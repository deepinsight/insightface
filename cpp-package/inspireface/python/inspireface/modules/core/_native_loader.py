"""ctypes library loading infrastructure for the native declarations."""

import ctypes
import ctypes.util
import os
import sys
from typing import Iterable, Iterator, List


class LibraryLoader:
    """Load a library and expose its symbols by calling convention."""

    name_formats = ("%s",)

    class Lookup:
        mode = ctypes.DEFAULT_MODE

        def __init__(self, path: str):
            self.access = {"cdecl": ctypes.CDLL(path, self.mode)}

        def get(self, name: str, calling_convention: str = "cdecl"):
            if calling_convention not in self.access:
                raise LookupError(
                    "Unknown calling convention {!r} for function {!r}".format(
                        calling_convention,
                        name,
                    )
                )
            return getattr(self.access[calling_convention], name)

        def has(self, name: str, calling_convention: str = "cdecl") -> bool:
            return calling_convention in self.access and hasattr(
                self.access[calling_convention],
                name,
            )

        def __getattr__(self, name: str):
            return getattr(self.access["cdecl"], name)

    def __init__(self) -> None:
        self.other_dirs: List[str] = []

    def __call__(self, libname: str):
        errors = []
        for path in self.getpaths(libname):
            try:
                return self.Lookup(path)
            except OSError as error:
                errors.append((path, error))

        message = "Could not load {}".format(libname)
        if errors:
            detail = "; ".join("{}: {}".format(path, error) for path, error in errors)
            raise ImportError("{}. Tried: {}".format(message, detail)) from errors[-1][1]
        raise ImportError(message)

    def getpaths(self, libname: str) -> Iterator[str]:
        if os.path.isabs(libname):
            yield libname
            return

        for directory in self.other_dirs:
            for name_format in self.name_formats:
                yield os.path.join(directory, name_format % libname)

        for name_format in self.name_formats:
            discovered = ctypes.util.find_library(name_format % libname)
            if discovered:
                yield discovered

        for name_format in self.name_formats:
            yield os.path.abspath(name_format % libname)


class DarwinLibraryLoader(LibraryLoader):
    name_formats = (
        "lib%s.dylib",
        "lib%s.so",
        "lib%s.bundle",
        "%s.dylib",
        "%s.so",
        "%s.bundle",
        "%s",
    )

    class Lookup(LibraryLoader.Lookup):
        mode = ctypes.RTLD_GLOBAL


class PosixLibraryLoader(LibraryLoader):
    name_formats = ("lib%s.so", "%s.so", "%s")


class WindowsLibraryLoader(LibraryLoader):
    name_formats = ("%s.dll", "lib%s.dll", "%slib.dll", "%s")

    class Lookup(LibraryLoader.Lookup):
        def __init__(self, path: str):
            super().__init__(path)
            self.access["stdcall"] = ctypes.windll.LoadLibrary(path)


_LOADER_TYPES = {
    "darwin": DarwinLibraryLoader,
    "cygwin": WindowsLibraryLoader,
    "win32": WindowsLibraryLoader,
    "msys": WindowsLibraryLoader,
}

load_library = _LOADER_TYPES.get(sys.platform, PosixLibraryLoader)()


def add_library_search_dirs(other_dirs: Iterable[str]) -> None:
    """Add directories used for non-absolute library names."""
    for path in other_dirs:
        normalized = os.path.abspath(path)
        if normalized not in load_library.other_dirs:
            load_library.other_dirs.append(normalized)
