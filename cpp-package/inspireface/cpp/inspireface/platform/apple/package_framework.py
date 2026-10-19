#!/usr/bin/env python3
"""Complete framework metadata and, on iOS, merge the private static dependency."""
import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "command/apple"))
from binary_info import inspect

p = argparse.ArgumentParser()
p.add_argument('--framework', type=Path, required=True)
p.add_argument('--modulemap', type=Path, required=True)
p.add_argument('--sdk', choices=['macosx', 'iphoneos', 'iphonesimulator'], required=True)
p.add_argument('--minimum', default='')
p.add_argument('--archive', type=Path)
p.add_argument('--coreml', action='store_true')
a = p.parse_args()
base = a.framework / 'Versions/A' if a.sdk == 'macosx' else a.framework
(base / 'Modules').mkdir(exist_ok=True)
shutil.copyfile(a.modulemap, base / 'Modules/module.modulemap')
if a.coreml:
    modulemap = base / 'Modules/module.modulemap'
    text = modulemap.read_text().replace('    export *', '    link framework "CoreML"\n    link framework "Accelerate"\n    export *', 1)
    modulemap.write_text(text)
if a.sdk == 'macosx':
    link = a.framework / 'Modules'
    if not link.is_symlink():
        link.symlink_to('Versions/Current/Modules')
else:
    if not a.archive:
        p.error('iOS static frameworks require a platform-specific MNN archive')
    binary = base / 'InspireFace'
    merged = base / '.InspireFace-merged'
    subprocess.run(['xcrun', 'libtool', '-static', '-o', str(merged), str(binary), str(a.archive)], check=True)
    merged.replace(binary)
    plist = base / 'Info.plist'
    data = plistlib.loads(plist.read_bytes())
    minimum = max((v['minimum'] for v in inspect(binary).values()), key=lambda v: tuple(map(int, v.split('.'))))
    data.update(MinimumOSVersion=minimum, CFBundleSupportedPlatforms=[
        'iPhoneOS' if a.sdk == 'iphoneos' else 'iPhoneSimulator'])
    plist.write_bytes(plistlib.dumps(data))
