#!/bin/bash
# Reuses the regular SDK build/cache; pass --run-simulator on a simulator-capable runner.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$repo_dir/command/apple/build_sdk.py" --verify --package "$@"
