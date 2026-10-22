#!/usr/bin/env python3
"""Assemble already-built CPU slices without compiling another SDK or including CoreML."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {('macosx', 'arm64'), ('macosx', 'x86_64'), ('iphoneos', 'arm64'),
            ('iphonesimulator', 'arm64'), ('iphonesimulator', 'x86_64')}


def cpu_sdks(directory, version=''):
    found = {}
    for metadata in directory.glob('*/sdk-info.json'):
        if version and not metadata.parent.name.endswith('-' + version):
            continue
        info = json.loads(metadata.read_text())
        if info['backend'] != 'cpu':
            continue
        key = (info['sdk'], info['arch'])
        if key in found:
            raise RuntimeError(f'Duplicate CPU SDK for {key}: {found[key]}, {metadata.parent}')
        found[key] = metadata.parent.resolve()
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root', type=Path, default=ROOT / 'build')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version', default='')
    p.add_argument('--scope', choices=['apple', 'ios'], default='apple')
    a = p.parse_args()
    expected = EXPECTED if a.scope == 'apple' else {pair for pair in EXPECTED if pair[0] != 'macosx'}
    sdks = {key: value for key, value in cpu_sdks(a.build_root, a.version).items() if key in expected}
    if set(sdks) != expected:
        raise RuntimeError(f'Missing CPU slices: {sorted(expected - sdks.keys())}')
    subprocess.run([sys.executable, str(ROOT / 'command/apple/package_xcframeworks.py'),
                    '--output', str(a.output), *map(str, sdks.values())], check=True)
    subprocess.run([sys.executable, str(ROOT / 'cpp/test/apple/verify_xcframeworks.py'),
                    '--package', str(a.output), '--output', str(a.build_root / 'apple-package-consumers'),
                    '--native-only'], check=True)


if __name__ == '__main__':
    main()
