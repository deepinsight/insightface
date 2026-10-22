# CI/CD

CI clones `https://github.com/tunmx/inspireface-3rdparty.git` with
`--recurse-submodules`. Dependency versions are selected by that repository's
submodule commits. InspireCV contains the compiler compatibility fixes upstream;
InspireFace does not apply a local dependency patch.

Build scripts stop at the first failure. Docker Compose jobs use
`--exit-code-from` so a failed container also fails its Actions step.

The manylinux2014 wheel scripts use Python 3.12 by default, with PEP 517 build
isolation for the dependencies in `python/pyproject.toml`. Override
`ISF_WHEEL_PYTHON` to select another installed interpreter. The ctypes package
produces one `py3-none` wheel per platform; building that wheel does not replace
testing installation and imports on each supported Python version.

The Ubuntu Python test script downloads `Pikachu` before building and passes
explicit fixture/model paths to `sample_testcase.run`. Images under
`test_res/data` are checked in; `test_res/pack` is ignored by Git and must be
populated on a fresh runner. Model downloads fail on HTTP/network errors and
replace the target file only after a successful, nonempty download.

Windows jobs use the `windows-2022` runner and the shared
`.github/actions/windows-setup` action. The action selects x64 Python and runs
`ci/windows/setup_environment.ps1` to initialize MSVC, CMake, Ninja, and the
recursive source dependencies. Compiler environment variables are carried to
later steps through `GITHUB_ENV`. Environment setup is separate from the local
`command/build_windows.ps1` and `command/build_wheel_windows.ps1` build entries.

Build Wheels produces five platform artifacts, including `windows-x64-wheels`.
The Windows wheel is installed in a fresh virtual environment and checked
against its bundled DLL before upload. The publication job requires one valid
wheel for every platform, including when reusing an earlier Actions run.
Pushes to `release/wheels` build all five platforms and publish
the validated collection to PyPI using the existing `PYPI_API_TOKEN` secret.
Pull requests targeting that branch build and validate without publishing.
Manual runs publish only when `publish` is enabled. Package versions come from
the CMake native version plus `python/post`; all five wheels must match that
version before publication.

`Test Windows PyPI Package` (`.github/workflows/windows-pypi-smoke.yaml`) runs on
pushes to `feature/win` that change its workflow, `ci/windows/verify_pypi_inference.py`,
or `test_res/data/bulk/kun.jpg`, and supports manual runs. It creates a fresh Python
3.13 environment, runs `pip install inspireface`, installs OpenCV for reading
the test image, and calls `inspireface.launch()` to automatically download and
cache the default Pikachu model through the Python package. It checks face
detection, dense landmarks, and feature extraction using the installed package.
The `windows-pypi-inference` artifact contains an annotated image, feature
vectors, and a JSON result summary. This workflow does not build or publish
wheels. A matching push runs the check before the workflow is merged into the
default branch. GitHub shows its **Run workflow** button after the workflow file
is present on the repository's default branch; a branch can then be selected
for a manual run.

`Build Windows SDK` (`.github/workflows/windows-sdk.yaml`) supports manual runs
and is reused by Build SDKs and Release SDKs. It has no direct push trigger.
The Windows SDK job builds the Release shared CPU SDK,
compiles and runs installed C and C++ consumers, and uploads
`sdk_files_windows_x64`. It records the exact InspireFace, third-party,
InspireCV, and MNN revisions in `windows-x64-build-diagnostics`, alongside
available CMake and CTest logs, including when a build fails. Update the
InspireCV submodule commit in `inspireface-3rdparty` before triggering a build
that needs a newer dependency version. Tagged releases include
`inspireface-windows-x64-<version>.zip` and its SHA-256 checksum. These packaging
jobs omit model-dependent native tests; running those tests locally requires
the resources described in `doc/Windows.md`.
