#!/usr/bin/env python3
"""Select every packaged XCFramework slice and compile real installed consumers."""
import argparse
import json
from pathlib import Path
import plistlib
import platform
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'command/apple'))
from binary_info import validate

p = argparse.ArgumentParser()
p.add_argument('--package', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--run-simulator', action='store_true')
p.add_argument('--native-only', action='store_true', help='Run only the host architecture; compile/link the other slices')
a = p.parse_args()
a.package = a.package.resolve()
a.output.mkdir(parents=True, exist_ok=True)
selected = {}
for name in ('InspireFace', 'InspireFaceSwift'):
    bundle = a.package / f'{name}.xcframework'
    metadata = plistlib.loads((bundle / 'Info.plist').read_bytes())
    for entry in metadata['AvailableLibraries']:
        sdk = 'macosx' if entry['SupportedPlatform'] == 'macos' else (
            'iphonesimulator' if entry.get('SupportedPlatformVariant') == 'simulator' else 'iphoneos')
        framework = bundle / entry['LibraryIdentifier'] / entry['LibraryPath']
        validate(framework / name, sdk, entry['SupportedArchitectures'])
        for arch in entry['SupportedArchitectures']:
            key = (sdk, arch)
            if name in selected.setdefault(key, {}):
                raise RuntimeError(f'Duplicate XCFramework slice for {name}/{key}')
            selected[key][name] = framework
expected = {(entry['sdk'], entry['arch']) for entry in json.loads((a.package / 'sdk-manifest.json').read_text())}
if set(selected) != expected:
    raise RuntimeError(f'XCFramework architecture set differs from SDK manifest: {selected.keys()}, {expected}')
for (sdk, arch), frameworks in selected.items():
    if set(frameworks) != {'InspireFace', 'InspireFaceSwift'}:
        raise RuntimeError(f'Core and Swift slices differ: {sdk}/{arch}')
    with tempfile.TemporaryDirectory(prefix=f'{sdk}-{arch}-', dir=a.output) as temp:
        temp = Path(temp)
        legacy = a.package / 'SDKs' / f'{sdk}-{arch}'
        shutil.copytree(legacy / 'InspireFace', temp / 'InspireFace', symlinks=True)
        if sdk != 'macosx':
            shutil.copytree(legacy / 'MNN.framework', temp / 'MNN.framework', symlinks=True)
        for name, framework in frameworks.items():
            shutil.copytree(framework, temp / f'{name}.framework', symlinks=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('verify_install.py')),
                        '--sdk', str(temp), '--output', str(a.output / f'{sdk}-{arch}'), '--arch', arch,
                        *(['--compile-only'] if a.native_only and arch != platform.machine() else []),
                        *(['--run-simulator'] if a.run_simulator and sdk == 'iphonesimulator' else [])], check=True)
print(f'XCFramework consumers validated: {len(selected)} platform/architecture combinations')
