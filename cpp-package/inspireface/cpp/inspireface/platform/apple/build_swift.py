#!/usr/bin/env python3
"""Build the Swift overlay without changing the repository's CMake generator."""
import argparse
import json
from pathlib import Path
import plistlib
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "command/apple"))
from binary_info import inspect


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--sdk', choices=['macosx', 'iphoneos', 'iphonesimulator'], default='macosx')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--deployment-target', default='')
    parser.add_argument('--architectures', default='')
    parser.add_argument('--sanitize-address', action='store_true')
    args = parser.parse_args()
    core = args.framework_dir / 'InspireFace.framework/InspireFace'
    architectures = run('xcrun', 'lipo', '-archs', str(core)).split()
    if args.architectures and set(args.architectures.split(';')) != set(architectures):
        raise RuntimeError('Swift overlay architectures must match the core framework')
    framework = args.framework_dir / 'InspireFaceSwift.framework'
    version = framework / 'Versions/A' if args.sdk == 'macosx' else framework
    modules = version / 'Modules/InspireFaceSwift.swiftmodule'
    resources = version / 'Resources' if args.sdk == 'macosx' else version
    modules.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    links = [('Versions/Current', 'A'), ('InspireFaceSwift', 'Versions/Current/InspireFaceSwift'),
                              ('Modules', 'Versions/Current/Modules'), ('Resources', 'Versions/Current/Resources')]
    for name, destination in links if args.sdk == 'macosx' else []:
        link = framework / name
        if not link.is_symlink():
            link.symlink_to(destination)
    with (resources / 'Info.plist').open('wb') as output:
        plistlib.dump(dict(CFBundleExecutable='InspireFaceSwift', CFBundleName='InspireFaceSwift',
                          CFBundleIdentifier='org.inspireface.sdk.swift', CFBundlePackageType='FMWK',
                          CFBundleVersion=args.version, CFBundleShortVersionString=args.version), output)
    sdk = run('xcrun', '--sdk', args.sdk, '--show-sdk-path')
    if args.sdk != 'macosx':
        plist = resources / 'Info.plist'
        data = plistlib.loads(plist.read_bytes())
        data.update(MinimumOSVersion=args.deployment_target, CFBundleSupportedPlatforms=[
            'iPhoneOS' if args.sdk == 'iphoneos' else 'iPhoneSimulator'])
        plist.write_bytes(plistlib.dumps(data))
    binaries = []
    for arch in architectures:
        minimum = args.deployment_target
        if not minimum:
            metadata = run('xcrun', 'vtool', '-arch', arch, '-show-build', str(core))
            match = re.search(r'\bminos\s+(\S+)', metadata)
            if not match:
                raise RuntimeError('Cannot determine the core framework deployment target')
            minimum = match[1]
        os_name = 'macosx' if args.sdk == 'macosx' else 'ios'
        suffix = '-simulator' if args.sdk == 'iphonesimulator' else ''
        target = f'{arch}-apple-{os_name}{minimum}{suffix}'
        info = json.loads(run('xcrun', '--sdk', args.sdk, 'swiftc', '-print-target-info', '-target', target))
        triple = info['target']['moduleTriple']
        binary = version / f'.InspireFaceSwift-{arch}'
        sanitizer = ['-sanitize=address'] if args.sanitize_address else []
        linkage = ['-static'] if args.static else [
            '-Xlinker', '-install_name', '-Xlinker',
            '@rpath/InspireFaceSwift.framework/Versions/A/InspireFaceSwift',
            '-Xlinker', '-current_version', '-Xlinker', args.version,
            '-Xlinker', '-compatibility_version', '-Xlinker', args.version.split('.')[0]]
        subprocess.run(['xcrun', '--sdk', args.sdk, 'swiftc', *sanitizer, '-swift-version', '5', '-O', '-whole-module-optimization',
                        '-sdk', sdk, '-target', target, '-parse-as-library', '-emit-library',
                        '-enable-library-evolution', '-emit-module', '-module-name', 'InspireFaceSwift',
                        '-emit-module-path', str(modules / f'{triple}.swiftmodule'),
                        '-emit-module-interface-path', str(modules / f'{triple}.swiftinterface'),
                        '-module-cache-path', str(args.framework_dir / '.module-cache'),
                        '-F', str(args.framework_dir), '-framework', 'InspireFace',
                        *linkage,
                        str(args.source), '-o', str(binary)], check=True)
        binaries.append(binary)
    output = version / 'InspireFaceSwift'
    if len(binaries) == 1:
        binaries[0].replace(output)
    else:
        subprocess.run(['xcrun', 'lipo', '-create', *map(str, binaries), '-output', str(output)], check=True)
        for binary in binaries:
            binary.unlink()
    if args.sdk != 'macosx':
        plist = resources / 'Info.plist'
        data = plistlib.loads(plist.read_bytes())
        data['MinimumOSVersion'] = max((v['minimum'] for v in inspect(output).values()), key=lambda v: tuple(map(int, v.split('.'))))
        plist.write_bytes(plistlib.dumps(data))


if __name__ == '__main__':
    main()
