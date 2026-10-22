#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRY_RUN=0
SKIP_TESTS=0
ALLOW_DIRTY=0
WITH_FACE3D=0
NO_CLEAN=0
SKIP_EXISTING_CHECK=0
REPOSITORY="pypi"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Build and upload the InsightFace package to official PyPI.

Usage:
  bash packaging/pypi/build_upload_pypi.sh [options]

Options:
  --dry-run              Run tests, build and validate packages. Do not upload.
  --skip-tests           Skip the local pytest release suite.
  --allow-dirty          Allow publishing from a dirty git working tree.
  --with-face3d          Include the optional face3d Cython/C++ extension.
  --no-clean             Keep existing build caches and dist files; upload only this build.
  --skip-existing-check  Skip the PyPI "version already exists" check.
  --repository NAME      Twine repository name. Defaults to "pypi".
  --python PATH          Python executable to use. Defaults to $PYTHON_BIN or python.
  -h, --help             Show this help message.

Credentials:
  Recommended token upload:
    export TWINE_USERNAME=__token__
    export TWINE_PASSWORD='pypi-...'

  Twine also supports ~/.pypirc, keyring, and PyPI Trusted Publisher in CI.

Examples:
  bash packaging/pypi/build_upload_pypi.sh --dry-run
  bash packaging/pypi/build_upload_pypi.sh
  bash packaging/pypi/build_upload_pypi.sh --with-face3d
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-tests)
      SKIP_TESTS=1
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      ;;
    --with-face3d)
      WITH_FACE3D=1
      ;;
    --no-clean)
      NO_CLEAN=1
      ;;
    --skip-existing-check)
      SKIP_EXISTING_CHECK=1
      ;;
    --repository)
      shift
      REPOSITORY="${1:-}"
      ;;
    --python)
      shift
      PYTHON_BIN="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python executable is empty." >&2
  exit 2
fi

if [[ "${REPOSITORY}" != "pypi" ]]; then
  echo "This script is intended for the official PyPI release path." >&2
  echo "Use --repository pypi, or create a separate TestPyPI script if needed." >&2
  exit 2
fi

cd "${PACKAGE_ROOT}"

echo "==> Package root: ${PACKAGE_ROOT}"
echo "==> Python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"

read -r PACKAGE_NAME PACKAGE_VERSION < <("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re

root = Path.cwd()
setup_text = (root / "setup.py").read_text(encoding="utf-8")
init_text = (root / "insightface" / "__init__.py").read_text(encoding="utf-8")

name_match = re.search(r"name\s*=\s*['\"]([^'\"]+)['\"]", setup_text)
version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text, re.M)

if not name_match:
    raise SystemExit("Unable to find package name in setup.py")
if not version_match:
    raise SystemExit("Unable to find __version__ in insightface/__init__.py")

print(name_match.group(1), version_match.group(1))
PY
)

echo "==> Package: ${PACKAGE_NAME} ${PACKAGE_VERSION}"

if [[ "${PACKAGE_NAME}" != "insightface" ]]; then
  echo "Refusing to upload package '${PACKAGE_NAME}' to official PyPI." >&2
  echo "Expected package name: insightface" >&2
  exit 1
fi

if [[ "${ALLOW_DIRTY}" -ne 1 ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "Git working tree is dirty. Commit or stash changes before release." >&2
      echo "Use --allow-dirty only for an intentional emergency release." >&2
      git status --short
      exit 1
    fi
  fi
fi

echo "==> Checking build tools"
"${PYTHON_BIN}" - "${SKIP_TESTS}" <<'PY'
import importlib.util
import sys

skip_tests = sys.argv[1] == "1"
required = ["build", "twine"]
if not skip_tests:
    required.append("pytest")

missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing required packaging tools: " + ", ".join(missing))
    print("Install them with: python -m pip install build twine pytest")
    raise SystemExit(1)
PY

if [[ "${SKIP_TESTS}" -ne 1 ]]; then
  echo "==> Running the complete release test suite"
  QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}" "${PYTHON_BIN}" -m pytest -q tests
else
  echo "==> Skipping tests"
fi

if [[ "${SKIP_EXISTING_CHECK}" -ne 1 ]]; then
  echo "==> Checking whether ${PACKAGE_NAME} ${PACKAGE_VERSION} already exists on PyPI"
"${PYTHON_BIN}" - "${PACKAGE_NAME}" "${PACKAGE_VERSION}" <<'PY'
import sys
import urllib.error
import urllib.request

name, version = sys.argv[1], sys.argv[2]
url = f"https://pypi.org/pypi/{name}/{version}/json"
try:
    with urllib.request.urlopen(url, timeout=20) as response:
        response.read(1)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        print("Version is not present on PyPI.")
    else:
        raise
else:
    print(f"{name} {version} already exists on PyPI. PyPI versions are immutable.")
    raise SystemExit(1)
PY
fi

if [[ "${NO_CLEAN}" -ne 1 ]]; then
  echo "==> Cleaning build artifacts"
  rm -rf build dist ./*.egg-info
fi

if [[ "${WITH_FACE3D}" -eq 1 ]]; then
  export INSIGHTFACE_WITH_FACE3D=1
  echo "==> face3d build: enabled"
else
  unset INSIGHTFACE_WITH_FACE3D || true
  echo "==> face3d build: disabled"
fi

RELEASE_WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/insightface-release.XXXXXXXX")"
trap 'rm -rf -- "${RELEASE_WORK_DIR}"' EXIT
RELEASE_DIST_DIR="${RELEASE_WORK_DIR}/dist"

echo "==> Building source distribution and wheel"
"${PYTHON_BIN}" -m build --outdir "${RELEASE_DIST_DIR}"

echo "==> Validating this build's package name, version and Python requirement"
"${PYTHON_BIN}" packaging/pypi/release_artifacts.py \
  --dist-dir "${RELEASE_DIST_DIR}" --name "${PACKAGE_NAME}" --version "${PACKAGE_VERSION}" \
  > "${RELEASE_WORK_DIR}/artifacts.list"
RELEASE_ARTIFACTS=()
while IFS= read -r -d '' artifact; do
  RELEASE_ARTIFACTS+=("${artifact}")
done < "${RELEASE_WORK_DIR}/artifacts.list"

INSPECT_OPTIONS=(inspect)
if [[ "${WITH_FACE3D}" -eq 1 ]]; then
  INSPECT_OPTIONS+=(--allow-native)
fi
"${PYTHON_BIN}" packaging/pypi/artifact_smoke.py "${INSPECT_OPTIONS[@]}" \
  --wheel "${RELEASE_ARTIFACTS[0]}" --sdist "${RELEASE_ARTIFACTS[1]}"

echo "==> Checking distributions"
"${PYTHON_BIN}" -m twine check --strict "${RELEASE_ARTIFACTS[@]}"

echo "==> Saving validated artifacts in dist/"
"${PYTHON_BIN}" - "${RELEASE_ARTIFACTS[@]}" <<'PY'
import os
from pathlib import Path
import shutil
import sys
import tempfile

destination = Path("dist")
destination.mkdir(exist_ok=True)
for source_name in sys.argv[1:]:
    source = Path(source_name)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
        target = destination / source.name
        os.replace(temporary, target)
        print(f"{target}: {source.stat().st_size:,} bytes")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete. No upload was performed."
  exit 0
fi

echo
echo "You are about to upload ${PACKAGE_NAME} ${PACKAGE_VERSION} to official PyPI."
echo "This cannot be overwritten or deleted for normal users after publishing."
echo
printf "Type exactly 'upload %s %s to pypi' to continue: " "${PACKAGE_NAME}" "${PACKAGE_VERSION}"
read -r CONFIRMATION
EXPECTED="upload ${PACKAGE_NAME} ${PACKAGE_VERSION} to pypi"
if [[ "${CONFIRMATION}" != "${EXPECTED}" ]]; then
  echo "Confirmation did not match. Upload canceled."
  exit 1
fi

echo "==> Uploading this build's validated artifacts to official PyPI"
"${PYTHON_BIN}" -m twine upload --repository "${REPOSITORY}" "${RELEASE_ARTIFACTS[@]}"

cat <<RELEASE_DONE

Upload complete.

Verify the release:
  python -m pip install --upgrade --no-cache-dir ${PACKAGE_NAME}==${PACKAGE_VERSION}
  python -c "import insightface; print(insightface.__version__)"

GUI extra:
  python -m pip install --upgrade --no-cache-dir '${PACKAGE_NAME}[gui]==${PACKAGE_VERSION}'
  insightface-gui --version
RELEASE_DONE
