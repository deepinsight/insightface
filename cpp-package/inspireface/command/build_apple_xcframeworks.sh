#!/bin/bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$repo_dir/command/apple/build_sdk.py" --platform all --backend all --package "$@"
