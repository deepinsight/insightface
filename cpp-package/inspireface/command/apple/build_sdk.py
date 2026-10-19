#!/usr/bin/env python3
"""Incremental Apple slices; the build cache is separate from distributable SDKs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
from binary_info import inspect

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def output(*args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def copy_directory(source, destination):
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=True)


def legacy_name(sdk, arch, backend):
    coreml = '-coreml' if backend == 'coreml' else ''
    if sdk == 'macosx':
        cpu = 'apple-silicon-arm64' if arch == 'arm64' else 'intel-x86-64'
        return f'inspireface-macos{coreml}-{cpu}'
    if sdk == 'iphoneos':
        return 'inspireface-ios-coreml-arm64' if backend == 'coreml' else 'inspireface-ios'
    return f'inspireface-ios{coreml}-simulator-{arch}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', choices=['macosx', 'iphoneos', 'iphonesimulator', 'ios', 'all'], default='all')
    p.add_argument('--arch', choices=['arm64', 'x86_64'])
    p.add_argument('--backend', choices=['cpu', 'coreml', 'all'], default='all')
    p.add_argument('--cache-root', type=Path, default=Path(os.environ.get('ISF_APPLE_CACHE_DIR', ROOT / 'build/apple-cache')))
    p.add_argument('--output-root', type=Path, default=ROOT / 'build')
    p.add_argument('--package', action='store_true')
    p.add_argument('--tests', action='store_true', help='Compile shared language contract tests for each slice')
    p.add_argument('--verify', action='store_true', help='Compile installed consumers and run executable host contracts')
    p.add_argument('--coverage', action='store_true', help='Verify macOS API execution coverage, then remove instrumentation before installation')
    p.add_argument('--run-simulator', action='store_true', help='Also run simulator contracts on ISF_APPLE_SIMULATOR or booted')
    p.add_argument('--jobs', type=int, default=int(os.environ.get('ISF_BUILD_JOBS', '4')))
    a = p.parse_args()
    a.tests = a.tests or a.verify
    if a.run_simulator and not a.verify:
        p.error('--run-simulator requires --verify')
    if a.coverage and (not a.verify or a.platform != 'macosx'):
        p.error('--coverage requires --verify --platform macosx')
    if a.verify:
        for fixture in (ROOT / 'test_res/pack/Pikachu', ROOT / 'test_res/data/bulk/kun.jpg'):
            if not fixture.is_file():
                p.error(f'Model verification requires {fixture}')
    a.cache_root = a.cache_root.resolve()
    a.output_root = a.output_root.resolve()
    a.output_root.mkdir(parents=True, exist_ok=True)
    platforms = ['macosx', 'iphoneos', 'iphonesimulator'] if a.platform == 'all' else (
        ['iphoneos', 'iphonesimulator'] if a.platform == 'ios' else [a.platform])
    backends = ['cpu', 'coreml'] if a.backend == 'all' else [a.backend]
    version = os.environ.get('VERSION', '')
    tag = '-' + version if version else ''
    toolchain = output('xcodebuild', '-version')
    if not (ROOT / '3rdparty').exists():
        run('git', 'clone', '--recurse-submodules', 'https://github.com/tunmx/inspireface-3rdparty.git', ROOT / '3rdparty')
    source = ROOT / '3rdparty/MNN'
    if not (source / 'CMakeLists.txt').is_file():
        run('git', '-C', ROOT / '3rdparty', 'submodule', 'update', '--init', '--recursive')
    revision = output('git', '-C', source, 'rev-parse', 'HEAD')
    defines = (source / 'include/MNN/MNNDefine.h').read_text()
    mnn_version = '.'.join(re.search(r'#define MNN_VERSION_' + part + r'\s+(\d+)', defines)[1]
                           for part in ('MAJOR', 'MINOR', 'PATCH'))
    manifests = {backend: [] for backend in backends}
    for sdk in platforms:
        arches = [a.arch] if a.arch else (['arm64'] if sdk == 'iphoneos' else ['arm64', 'x86_64'])
        if sdk == 'iphoneos' and arches != ['arm64']:
            p.error('iOS device supports arm64; x86_64 requires iphonesimulator')
        sdk_path = output('xcrun', '--sdk', sdk, '--show-sdk-path')
        minimum = os.environ.get('IOS_DEPLOYMENT_TARGET', '11.0') if sdk != 'macosx' else os.environ.get('MACOSX_DEPLOYMENT_TARGET', '')
        for arch in arches:
            common = [f'-DCMAKE_OSX_SYSROOT={sdk_path}', f'-DCMAKE_OSX_ARCHITECTURES={arch}',
                      f'-DCMAKE_SYSTEM_PROCESSOR={arch}', '-DCMAKE_BUILD_TYPE=Release',
                      '-DCMAKE_POLICY_VERSION_MINIMUM=3.5']
            if minimum:
                common.append(f'-DCMAKE_OSX_DEPLOYMENT_TARGET={minimum}')
            if sdk != 'macosx':
                common += ['-DCMAKE_SYSTEM_NAME=iOS', '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY']
            dependency_options = ['-DMNN_BUILD_SHARED_LIBS=OFF', '-DMNN_BUILD_TOOLS=OFF',
                                  '-DMNN_BUILD_DEMO=OFF', '-DMNN_SEP_BUILD=OFF',
                                  '-DMNN_METAL=OFF', '-DMNN_COREML=OFF', '-DMNN_OPENCL=OFF', '-DMNN_VULKAN=OFF']
            if arch == 'x86_64':
                # MNN 2.8.x cpu_id.cc relies on transitive fixed-width integer declarations.
                dependency_options.append('-DCMAKE_CXX_FLAGS=-include cstdint')
            fingerprint = hashlib.sha256(json.dumps([toolchain, sdk_path, revision, common, dependency_options]).encode()).hexdigest()[:16]
            dependency = a.cache_root / 'mnn' / f'{sdk}-{arch}-{fingerprint}'
            run('cmake', '-S', source, '-B', dependency, *common, *dependency_options)
            run('cmake', '--build', dependency, '--target', 'MNN', '--parallel', a.jobs)
            dep_sdk = dependency / 'sdk'
            (dep_sdk / 'lib').mkdir(parents=True, exist_ok=True)
            archive = dependency / 'libMNN.a'
            # Cache links are private. Installation resolves them to ordinary files.
            for link, target in [(dep_sdk / 'include', source / 'include'), (dep_sdk / 'lib/libMNN.a', archive)]:
                if not link.is_symlink():
                    link.symlink_to(target, target_is_directory=target.is_dir())
            for backend in backends:
                key = f'{sdk}-{arch}-{backend}-{fingerprint}'
                build = a.cache_root / 'sdk' / key
                static = sdk != 'macosx' or (backend == 'coreml' and arch == 'arm64')
                run('cmake', '-S', ROOT, '-B', build, *common,
                    f'-DMNN_STATIC_PATH={dep_sdk}', '-DISF_BUILD_APPLE_FRAMEWORK=ON',
                    # Apply SDK defaults to existing incremental caches as well.
                    '-DISF_ENABLE_INSPIRECV_TASK_PREPROCESS=ON', '-DINSPIRECV_TASK_ENABLE_ARM_NEON=ON',
                    f'-DISF_ENABLE_APPLE_EXTENSION={"ON" if backend == "coreml" else "OFF"}',
                    f'-DISF_BUILD_SHARED_LIBS={"OFF" if static else "ON"}',
                    f'-DISF_BUILD_APPLE_TESTS={"ON" if a.tests else "OFF"}',
                    f'-DISF_APPLE_ENABLE_COVERAGE={"ON" if a.coverage else "OFF"}',
                    '-DISF_BUILD_WITH_TEST=OFF', '-DISF_BUILD_WITH_SAMPLE=OFF')
                run('cmake', '--build', build, '--parallel', a.jobs)
                if a.coverage:
                    profiles = build / 'apple-tests/profiles'
                    for profile in profiles.rglob('*.profraw'):
                        profile.unlink()
                    run('ctest', '--test-dir', build, '--output-on-failure', '-R', '^Apple\\.')
                    run(sys.executable, ROOT / 'cpp/test/apple/verify_api_coverage.py',
                        '--framework', build / 'apple/InspireFace.framework/InspireFace', '--profiles', profiles)
                    # Only the thin adapter and final links change; core objects and MNN are reused.
                    run('cmake', '-S', ROOT, '-B', build, '-DISF_APPLE_ENABLE_COVERAGE=OFF')
                    run('cmake', '--build', build, '--parallel', a.jobs)
                run('cmake', '--install', build)
                staging = a.output_root / (legacy_name(sdk, arch, backend) + tag)
                staging.mkdir(parents=True, exist_ok=True)
                for item in (build / 'install').iterdir():
                    destination = staging / item.name
                    if item.is_dir():
                        copy_directory(item, destination)
                    else:
                        shutil.copy2(item, destination)
                if sdk != 'macosx':
                    # Keep the original iOS MNN.framework consumer route as well.
                    mnn = staging / 'MNN.framework'
                    mnn.mkdir(exist_ok=True)
                    shutil.copy2(archive, mnn / 'MNN')
                    copy_directory(source / 'include/MNN', mnn / 'Headers')
                    (mnn / 'Info.plist').write_bytes(plistlib.dumps(dict(
                        CFBundleExecutable='MNN', CFBundleName='MNN', CFBundlePackageType='FMWK',
                        CFBundleIdentifier='org.inspireface.dependency.mnn', CFBundleVersion=mnn_version,
                        CFBundleShortVersionString=mnn_version, MinimumOSVersion=inspect(archive)[arch]['minimum'],
                        CFBundleSupportedPlatforms=['iPhoneOS' if sdk == 'iphoneos' else 'iPhoneSimulator'])))
                info = dict(sdk=sdk, arch=arch, backend=backend, minimum=minimum,
                            deployment=inspect(staging / 'InspireFace.framework/InspireFace'),
                            dependencyRevision=revision, dependencyKey=fingerprint, toolchain=toolchain)
                (staging / 'sdk-info.json').write_text(json.dumps(info, indent=2) + '\n')
                if a.verify:
                    run(sys.executable, ROOT / 'cpp/test/apple/verify_install.py', '--sdk', staging,
                        '--output', build / 'apple-tests/installed-consumers', '--arch', arch,
                        *(['--run-simulator'] if a.run_simulator and sdk == 'iphonesimulator' else []))
                    pattern = '^Apple\\.' if sdk == 'macosx' or (sdk == 'iphonesimulator' and a.run_simulator) else '^Apple.API.Mapping$'
                    if not a.coverage:
                        run('ctest', '--test-dir', build, '--output-on-failure', '-R', pattern)
                    if sdk != 'macosx' and not (sdk == 'iphonesimulator' and a.run_simulator):
                        print(f'{sdk}/{arch}: compiled and linked; runtime execution not requested', flush=True)
                manifests[backend].append(str(staging))
                print(f'SDK ready: {staging}', flush=True)
    if a.package:
        for backend, sdks in manifests.items():
            destination = a.output_root / ('inspireface-apple' + ('-coreml' if backend == 'coreml' else '') + tag)
            run(sys.executable, ROOT / 'command/apple/package_xcframeworks.py', '--output', destination, *sdks)
            if a.verify:
                run(sys.executable, ROOT / 'cpp/test/apple/verify_xcframeworks.py', '--package', destination,
                    '--output', a.cache_root / 'package-tests' / backend,
                    *(['--run-simulator'] if a.run_simulator else []))


if __name__ == '__main__':
    main()
