#!/usr/bin/env python3
"""Create the single CPU SDK release archive after checking its complete contents."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import stat
import subprocess
import zipfile

EXPECTED = {('macosx', 'arm64'), ('macosx', 'x86_64'), ('iphoneos', 'arm64'),
            ('iphonesimulator', 'arm64'), ('iphonesimulator', 'x86_64')}
TOP_LEVEL = {'InspireFace.xcframework', 'InspireFaceSwift.xcframework', 'Frameworks', 'SDKs', 'sdk-manifest.json'}


def finder_metadata(path):
    # Inspecting an SDK in Finder must not affect its distributable contents.
    return path.name == '.DS_Store'


def check_manifest(manifest, scope='apple'):
    expected = EXPECTED if scope == 'apple' else {pair for pair in EXPECTED if pair[0] != 'macosx'}
    if not manifest or any(entry.get('backend') != 'cpu' for entry in manifest):
        raise ValueError('Release accepts CPU SDKs only; CoreML must remain unpublished')
    actual = [(entry['sdk'], entry['arch']) for entry in manifest]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f'Release slice set is incomplete or duplicated: {actual}; expected {sorted(expected)}')
    return expected


def check_package(package, scope='apple'):
    package = package.resolve()
    expected = check_manifest(json.loads((package / 'sdk-manifest.json').read_text()), scope)
    if {item.name for item in package.iterdir() if not finder_metadata(item)} != TOP_LEVEL:
        raise ValueError('Unexpected or missing top-level release content')
    for path in package.rglob('*'):
        if finder_metadata(path):
            continue
        relative = path.relative_to(package)
        if ('prompt_docs' in relative.parts or path.suffix in {'.md', '.log', '.profraw', '.profdata'}
                or any(part.startswith('.') for part in relative.parts) or 'coreml' in str(relative).lower()):
            raise ValueError(f'Internal documents, test output or CoreML payload in release: {relative}')
        if path.is_symlink() and package not in path.resolve(strict=True).parents:
            raise ValueError(f'Release symlink escapes the package: {relative}')
    for sdk, arch in expected:
        root = package / 'SDKs' / f'{sdk}-{arch}'
        check = json.loads((root / 'sdk-info.json').read_text())
        if (check.get('backend'), check.get('sdk'), check.get('arch')) != ('cpu', sdk, arch):
            raise ValueError(f'Incorrect legacy SDK identity: {root}')
        if not (root / 'InspireFace/include/inspireface.h').is_file():
            raise ValueError(f'Missing legacy C headers: {root}')
        extension = 'dylib' if sdk == 'macosx' else 'a'
        if not (root / f'InspireFace/lib/libInspireFace.{extension}').is_file():
            raise ValueError(f'Missing legacy library: {root}')
    for name in ('InspireFace', 'InspireFaceSwift'):
        bundle = package / f'{name}.xcframework'
        libraries = plistlib.loads((bundle / 'Info.plist').read_bytes())['AvailableLibraries']
        actual = []
        for entry in libraries:
            sdk = 'macosx' if entry['SupportedPlatform'] == 'macos' else (
                'iphonesimulator' if entry.get('SupportedPlatformVariant') == 'simulator' else 'iphoneos')
            actual.extend((sdk, arch) for arch in entry['SupportedArchitectures'])
            binary = bundle / entry['LibraryIdentifier'] / entry['LibraryPath'] / name
            if name == 'InspireFace':
                symbols = subprocess.check_output(['xcrun', 'nm', '-j', str(binary)], text=True, stderr=subprocess.PIPE)
                if 'OBJC_CLASS_$_CoreMLAdapterImpl' in symbols or 'llvm_profile' in symbols:
                    raise ValueError(f'CoreML implementation or coverage instrumentation found in release: {binary}')
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'XCFramework slice set differs from CPU release manifest: {name}')
    return package


def write_archive(package, destination, root_name):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(package.rglob('*')):
                if finder_metadata(path):
                    continue
                name = (Path(root_name) / path.relative_to(package)).as_posix()
                if path.is_symlink():
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(info, os.readlink(path))
                elif path.is_file():
                    archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Release zip checksum validation failed')
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--output-dir', type=Path, default=Path('.'))
    p.add_argument('--scope', choices=['apple', 'ios'], default='apple')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+_-]*', a.version):
        p.error('Invalid release version')
    package = check_package(a.package, a.scope)
    root_name = f'inspireface-{a.scope}-{a.version}'
    destination = a.output_dir.resolve() / f'{root_name}.zip'
    write_archive(package, destination, root_name)
    print(f'CPU release archive ready: {destination}')


if __name__ == '__main__':
    main()
