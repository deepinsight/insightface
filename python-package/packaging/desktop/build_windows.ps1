$ErrorActionPreference = "Stop"

Write-Host "Building InsightFace Evaluation Studio for Windows..."
python -m PyInstaller --noconfirm packaging/desktop/pyinstaller.spec
Write-Host "Build complete. Output: dist/InsightFace Evaluation Studio/"
