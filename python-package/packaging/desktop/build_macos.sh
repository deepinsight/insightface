#!/usr/bin/env bash
set -euo pipefail

echo "Building InsightFace Evaluation Studio for macOS..."
python -m PyInstaller --noconfirm packaging/desktop/pyinstaller.spec
echo "Build complete. Output: dist/InsightFace Evaluation Studio.app or dist/InsightFace Evaluation Studio/"
