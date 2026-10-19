#!/usr/bin/env python3
"""Compile the shared Swift contracts for the selected Apple slice."""
import argparse
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--framework-dir', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sdk', default='macosx')
p.add_argument('--arch', default='')
p.add_argument('--minimum', default='')
p.add_argument('--sanitize-address', action='store_true')
p.add_argument('--coreml', action='store_true')
a = p.parse_args()
core = a.framework_dir / 'InspireFace.framework/InspireFace'
arch = a.arch or subprocess.check_output(['xcrun', 'lipo', '-archs', str(core)], text=True).split()[0]
minimum = a.minimum
if not minimum:
    metadata = subprocess.check_output(['xcrun', 'vtool', '-arch', arch, '-show-build', str(core)], text=True)
    minimum = re.search(r'\bminos\s+(\S+)', metadata).group(1)
os_name = 'macosx' if a.sdk == 'macosx' else 'ios'
suffix = '-simulator' if a.sdk == 'iphonesimulator' else ''
sdk = subprocess.check_output(['xcrun', '--sdk', a.sdk, '--show-sdk-path'], text=True).strip()
extra = (['-sanitize=address'] if a.sanitize_address else []) + (['-DIF_TEST_COREML'] if a.coreml else [])
subprocess.run(['xcrun', '--sdk', a.sdk, 'swiftc', *extra, '-O', '-swift-version', '5', '-target', f'{arch}-apple-{os_name}{minimum}{suffix}',
                '-sdk', sdk, '-module-cache-path', str(a.output.parent / '.module-cache'), '-F', str(a.framework_dir),
                '-framework', 'InspireFaceSwift', '-framework', 'InspireFace',
                '-framework', 'Foundation', '-framework', 'CoreVideo', '-framework', 'CoreML', '-framework', 'Accelerate',
                '-Xlinker', '-ObjC', '-lc++', '-L', str(a.output.parent), '-lAppleAllocationProbe',
                '-Xlinker', '-rpath', '-Xlinker', str(a.output.parent.resolve()),
                '-Xlinker', '-rpath', '-Xlinker', str(a.framework_dir.resolve()), str(a.source), '-o', str(a.output)], check=True)
