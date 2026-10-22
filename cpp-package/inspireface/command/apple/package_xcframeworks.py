#!/usr/bin/env python3
"""Combine same-platform architectures, then package separate backend XCFrameworks."""
import argparse
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
from binary_info import validate

NAMES = ('InspireFace', 'InspireFaceSwift')


def merge_framework(name, sdks, destination, platform):
    frameworks = [sdk / f'{name}.framework' for sdk in sdks]
    arches = [json.loads((sdk / 'sdk-info.json').read_text())['arch'] for sdk in sdks]
    if len(arches) != len(set(arches)):
        raise RuntimeError(f'Duplicate platform architecture: {platform}: {arches}')
    for framework, arch in zip(frameworks, arches):
        validate(framework / name, platform, [arch])
    shutil.copytree(frameworks[0], destination, symlinks=True)
    base = Path('Versions/A') if platform == 'macosx' else Path('.')
    # Public headers/module maps must describe the same API in every architecture.
    for framework in frameworks[1:]:
        for item in (framework / base).rglob('*'):
            relative = item.relative_to(framework / base)
            if item.is_file() and relative.parts[0] in ('Headers', 'Modules'):
                target = destination / base / relative
                if target.exists() and target.read_bytes() != item.read_bytes():
                    raise RuntimeError(f'Architecture-specific public interface collision: {relative}')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    if len(frameworks) > 1:
        subprocess.run(['xcrun', 'lipo', '-create', *[str(f / name) for f in frameworks],
                        '-output', str(destination / base / name)], check=True)
    info = validate(destination / name, platform, arches)
    if platform != 'macosx':
        plist = destination / 'Info.plist'
        data = plistlib.loads(plist.read_bytes())
        # Simulator arm64 starts at iOS 14; retain the lower x86_64 deployment target.
        data['MinimumOSVersion'] = min((v['minimum'] for v in info.values()), key=lambda v: tuple(map(int, v.split('.'))))
        plist.write_bytes(plistlib.dumps(data))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('sdks', type=Path, nargs='+')
    a = p.parse_args()
    sdks = [sdk.resolve() for sdk in a.sdks]
    infos = [json.loads((sdk / 'sdk-info.json').read_text()) for sdk in sdks]
    backends = {i['backend'] for i in infos}
    if len(backends) != 1:
        p.error('CPU and CoreML builds must be packaged separately')
    if len({(i['dependencyRevision'], i['toolchain']) for i in infos}) != 1:
        p.error('XCFramework slices must use the same MNN revision and Xcode toolchain')
    groups = {}
    for sdk, info in zip(sdks, infos):
        groups.setdefault(info['sdk'], []).append(sdk)
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build and validate in a sibling staging directory before replacing managed output.
    with tempfile.TemporaryDirectory(prefix='apple-package-', dir=output.parent) as temp:
        stage = Path(temp)
        for name in NAMES:
            inputs = []
            for platform, members in groups.items():
                destination = stage / 'Frameworks' / platform / f'{name}.framework'
                merge_framework(name, members, destination, platform)
                inputs += ['-framework', str(destination)]
            subprocess.run(['xcodebuild', '-create-xcframework', *inputs, '-output', str(stage / f'{name}.xcframework')], check=True)
            metadata = plistlib.loads((stage / f'{name}.xcframework/Info.plist').read_bytes())
            if len(metadata['AvailableLibraries']) != len(groups):
                raise RuntimeError('XCFramework has an incorrect number of platform slices')
        # Preserve architecture-specific legacy directories alongside the universal frameworks.
        for sdk, info in zip(sdks, infos):
            shutil.copytree(sdk, stage / 'SDKs' / f'{info["sdk"]}-{info["arch"]}', symlinks=True)
        (stage / 'sdk-manifest.json').write_text(json.dumps(infos, indent=2) + '\n')
        output.mkdir(exist_ok=True)
        for item in stage.iterdir():
            destination = output / item.name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            shutil.move(str(item), destination)
    print(f'XCFrameworks ready: {output}')


if __name__ == '__main__':
    main()
