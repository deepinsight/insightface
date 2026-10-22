#!/bin/bash
# Build native slices; only macOS arm64 executes SDK tests in Actions.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
backend="${1:-cpu}"
case "$backend" in cpu|coreml|all) ;; *) echo "Invalid Apple backend: $backend" >&2; exit 2 ;; esac
host_arch="$(uname -m)"
case "$host_arch" in arm64|x86_64) ;; *) echo "Unsupported Apple runner architecture: $host_arch" >&2; exit 2 ;; esac
macos_args=(--platform macosx --arch "$host_arch" --backend "$backend")
if [[ "$host_arch" == arm64 ]]; then
  macos_args+=(--verify --coverage)
fi
python3 "$repo_dir/command/apple/build_sdk.py" "${macos_args[@]}"
if [[ "$host_arch" == arm64 ]]; then
  python3 "$repo_dir/command/apple/build_sdk.py" --platform iphoneos --backend "$backend"
fi
python3 "$repo_dir/command/apple/build_sdk.py" --platform iphonesimulator --arch "$host_arch" \
  --backend "$backend"
