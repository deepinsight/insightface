#!/usr/bin/env bash
# Native JVM SDK: a shared Java 8-compatible JAR plus this target's JNI libraries.
# Example: ISF_JAVA_TESTS=ON bash command/build_java.sh -DMNN_STATIC_PATH=/path/to/mnn
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${ISF_JAVA_BUILD_DIR:-$repo_dir/build/java-sdk}"
cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DISF_BUILD_JAVA=ON \
    -DISF_BUILD_JAVA_TESTS="${ISF_JAVA_TESTS:-OFF}" \
    -DISF_BUILD_WITH_TEST=OFF \
    -DISF_BUILD_WITH_SAMPLE=OFF \
    "$@"
cmake --build "$build_dir" --parallel "${ISF_BUILD_JOBS:-4}"
ctest --test-dir "$build_dir" --output-on-failure -R '^InspireFace.Java\.' --no-tests=ignore
cmake --install "$build_dir"
echo "Java SDK ready: $build_dir/install/Java"
