#!/usr/bin/env python3
"""Create a legacy compact V1 manifest for Buffalo-style Server bundles.

Model identity is defined by its name; no separate model version is emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--detector", default="det_10g.onnx")
    parser.add_argument("--recognizer", default="w600k_r50.onnx")
    parser.add_argument("--model-id", default="buffalo_l")
    parser.add_argument("--display-name", default="Buffalo_L")
    parser.add_argument("--license", default="MODEL.LICENSE")
    parser.add_argument("--output", default="manifest.json")
    args = parser.parse_args()
    detector = (args.models_dir / args.detector).resolve()
    recognizer = (args.models_dir / args.recognizer).resolve()
    root = args.models_dir.resolve()
    license_path = (args.models_dir / args.license).resolve()
    for package_file in (detector, recognizer, license_path):
        if package_file.parent != root or not package_file.is_file():
            parser.error(
                f"package file does not exist directly in --models-dir: {package_file}"
            )
    manifest = {
        "manifest_version": 1,
        "model_id": args.model_id,
        "display_name": args.display_name,
        "files": {
            "detector": detector.name,
            "recognizer": recognizer.name,
        },
        "recognition": {
            "input_size": [112, 112],
            "embedding_dimension": 512,
            "preprocessing": "insightface-arcface-1",
        },
        "license": license_path.name,
    }
    output = root / args.output
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
