# Windows x64 CPU build

The Windows entry point builds the existing C/C++ SDK with the MNN CPU backend
and the InspireCV object library. Use Visual Studio 2022 C++ Build Tools, a
Windows SDK, CMake 3.20 or newer, Ninja, and Git. Open **x64 Native Tools Command
Prompt for VS 2022**, then start PowerShell. The script discovers Visual Studio's
bundled CMake and Ninja when those components are installed.

Keep the normal repository layout and initialize `3rdparty` as described in the
project build instructions. Its MNN, SQLite, yaml-cpp, and other source
dependencies are built through the existing CMake project. The build uses the
InspireCV version pinned by `3rdparty`.

```powershell
# From the InspireFace repository.
.\command\build_windows.ps1
.\command\build_windows.ps1 -Static
.\command\build_windows.ps1 -Configuration Debug
```

For local InspireCV development, `-InspireCVSource <checkout>` sets the optional
`ISF_INSPIRECV_SOURCE_DIR` override.

The default is Release, a shared SDK, and the complete registered CTest suite.
`-Samples` also builds the existing samples. `-Jobs 2` limits build parallelism.
`-SkipTests` is available for packaging without test resources; it is not a test
pass. Use separate build directories for Debug/Release and shared/static builds
so compiler flags, CRT libraries, and dependency artifacts stay consistent.
Run samples from a writable directory: Windows sample guards place temporary
archives and databases in the current directory when no explicit location is
provided. `TMPDIR`, when set for archive guards, must contain a UTF-8 path.

## Test resources

Use the project's existing `test_res/data` images, including the video frame
fixtures, and `test_res/pack/Pikachu`. Models and some fixtures are not tracked by
Git, so a source checkout alone may not contain everything needed by the tests.
The existing `command/download_models_general.sh Pikachu` downloads the model
from the project release. The `ci/quick_test_local.sh` resource URL supplies the
test image archive. The Windows build script creates the writable
`test_res/save/video_frames` output directory and runs all CTest entries.

## Installed C/C++ consumers

The original SDK layout remains `build/.../install/InspireFace/include` and
`build/.../install/InspireFace/lib`. On Windows the DLL is `libInspireFace.dll`,
matching the Python loader. The import library is separate from the DLL.

Use the installed CMake package so DLL import definitions and static MNN
dependencies propagate to the consumer:

```cmake
find_package(InspireFace CONFIG REQUIRED)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE InspireFace::InspireFace)
```

Configure with `-DInspireFace_DIR=<install-root>/InspireFace/lib/cmake/InspireFace`.
The installed package uses paths relative to its own location and can be moved
with the SDK. Match the consumer's configuration and CRT to the SDK build. Place
the DLL beside the application executable, or explicitly provide its directory
to the process's DLL search path.

Release builds use the dynamic MSVC runtime. Deploy the Microsoft Visual C++
2022 x64 Redistributable with the application. Debug builds require the matching
Visual Studio debug runtime and are intended for development; the redistributable
does not provide that runtime. A build machine with Visual Studio installed does
not establish that a separate consumer PC has the required runtime.

For a consumer that does not use CMake, include the installed headers, link the
installed import library, and define `ISF_BUILD_SHARED_LIBS` for the shared SDK.
C++ consumers of the embedded InspireCV API also need
`INSPIRECV_API=__declspec(dllimport)`. Static consumers must omit both DLL import
definitions and link the installed `MNN.lib` as well as `libInspireFace.lib`.

## Python and platform scope

Build a Windows wheel from an x64 Native Tools environment using x64 Python:

```powershell
.\command\build_wheel_windows.ps1 -PythonExecutable python -Jobs 2
```

This uses the existing native build entry point to build and install a Release
shared CPU/MNN SDK, then creates one
`inspireface-<version>-py3-none-win_amd64.whl` in `python/dist`. The ctypes wrapper
has no CPython extension ABI, so the wheel uses `py3-none`; the native DLL makes
it specific to Windows x64. The version comes from the project CMake version
and `python/post`, including when `python/version.txt` has not been generated.

An already built SDK can be packaged on a Windows x64 Python machine without
Visual Studio. Pass the install root containing `InspireFace/` and `version.txt`:

```powershell
.\command\build_wheel_windows.ps1 -PythonExecutable .\venv\Scripts\python.exe `
    -SdkDirectory 'C:\SDKs\InspireFace Windows' `
    -OutputDirectory .\python\dist
```

Packaging uses a fresh temporary Python project and puts only the selected DLL
at `inspireface/modules/core/libs/windows/x64/libInspireFace.dll`. Existing
`python/inspireface/modules/core/libs`, build caches, and wheels for other
platforms are left in place. The script checks SDK version, AMD64 PE format,
Release runtime imports, wheel metadata, and the exact bundled DLL hash. It
rejects Debug SDKs, x86/ARM64 libraries, and SDKs requiring additional native
DLLs such as shared MNN, OpenCV, or GPU runtimes. Additional DLLs are not copied
automatically. Deploy the VC++ x64 Redistributable described above separately.

`-BuildDirectory` selects a native build directory when no SDK is supplied;
`-OutputDirectory` selects the wheel destination. Native packaging builds omit
tests by default, as the Unix wheel scripts do. Add `-TestNative` to run the
complete native suite with prepared test resources. `-SdkDirectory` does not
rebuild or run native tests. PEP 517 installs build dependencies in isolation;
`-NoBuildIsolation` is available when the selected Python environment already
contains the requirements from `python/pyproject.toml`. `-Force` replaces an
existing wheel with the same filename. Each successful build also writes a
`.manifest.json` containing wheel/DLL hashes and imported runtime DLLs.

Install the printed wheel path with `python -m pip install <wheel-path>` and
import `inspireface` normally. Models remain separate resources; this wheel
contains the Python package and native runtime. The script only builds local
artifacts and does not publish them.

For local development, set `INSPIREFACE_LIBRARY_PATH` to the full installed
`libInspireFace.dll` path before importing the existing Python package. Normal
package loading uses `libs/windows/x64/libInspireFace.dll`. The native runtime,
Python process, and SDK must all use x64.

This entry point covers the x64 CPU/MNN path with MNN built as a static dependency.
Shared MNN builds, CUDA, TensorRT, and other optional
backends require their own dependencies and validation.
