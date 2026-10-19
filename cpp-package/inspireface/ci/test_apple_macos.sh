#!/bin/bash
# Local and CI entry point: public C/C++, Objective-C, and Swift consumers.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${ISF_APPLE_BUILD_DIR:-$repo_dir/build/apple-macos-tests}"
model_path="${ISF_APPLE_TEST_MODEL:-$repo_dir/test_res/pack/Pikachu}"
if [[ ! -f "$model_path" ]]; then
    echo "Missing test model: $model_path (set ISF_APPLE_TEST_MODEL)" >&2
    exit 1
fi
cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DISF_BUILD_APPLE_FRAMEWORK=ON \
    -DISF_BUILD_APPLE_TESTS=ON \
    -DISF_ENABLE_INSPIRECV_TASK_PREPROCESS=ON \
    -DINSPIRECV_TASK_ENABLE_ARM_NEON=ON \
    -DISF_APPLE_ENABLE_COVERAGE=ON \
    -DISF_BUILD_WITH_TEST=ON \
    -DISF_BUILD_WITH_SAMPLE=OFF \
    -DISF_ENABLE_APPLE_EXTENSION="${ISF_APPLE_COREML:-OFF}" \
    -DISF_APPLE_TEST_MODEL="$model_path" \
    "$@"
cmake --build "$build_dir" --parallel "${ISF_BUILD_JOBS:-4}"
# Remove previous execution profiles so stale runs cannot satisfy coverage.
python3 - "$build_dir/apple-tests/profiles" <<'PY'
from pathlib import Path
import sys
for path in Path(sys.argv[1]).rglob('*.profraw'):
    path.unlink()
PY
ctest --test-dir "$build_dir" --output-on-failure -R '^(Apple\.|InspireFace\.(CAPI|CPPAPI)\.)'
python3 "$repo_dir/cpp/test/apple/verify_api_coverage.py" \
    --framework "$build_dir/apple/InspireFace.framework/InspireFace" \
    --profiles "$build_dir/apple-tests/profiles"
cmake --install "$build_dir"
python3 "$repo_dir/cpp/test/apple/verify_install.py" \
    --sdk "$build_dir/install" --output "$build_dir/apple-tests/installed-consumers"
