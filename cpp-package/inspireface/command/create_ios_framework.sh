#!/bin/bash
# Compatibility entry point; framework generation now belongs to the shared build.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_dir/command/build_ios.sh" "$@"
