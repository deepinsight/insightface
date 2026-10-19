#!/usr/bin/env python3
"""Check and package the HarmonyOS C/C++ SDK and source HAR with prebuilt native code."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
NAPI_LIBRARY = 'src/main/libs/arm64-v8a/libinspireface_napi.so'
TYPE_MANIFEST = 'src/main/cpp/types/libinspireface_napi/oh-package.json5'


def check_version(native_install, har, requested):
    match = re.fullmatch(r'InspireFace Version: (\d+\.\d+\.\d+)\s*',
                         (native_install / 'version.txt').read_text())
    if not match:
        raise ValueError('Native SDK version.txt is missing a valid version')
    version = match[1]
    for manifest in (har / 'oh-package.json5', har / TYPE_MANIFEST):
        if json.loads(manifest.read_text()).get('version') != version:
            raise ValueError(f'HAR version does not match native SDK {version}: {manifest}')
    requested = requested.removeprefix('v') or version
    release = re.fullmatch(r'(\d+\.\d+\.\d+)(?:[-.][A-Za-z0-9][A-Za-z0-9.-]*)?', requested)
    if not release or release[1] != version:
        raise ValueError(f'Release version {requested!r} does not match native SDK {version}')
    return requested


def check_elf(output, napi=False):
    if not re.search(r'Machine:\s+AArch64\b', output):
        raise ValueError('HarmonyOS library must be AArch64')
    dependencies = set(re.findall(r'Shared library: \[(.+?)\]', output))
    expected = {'libc.so', 'libace_napi.z.so'} if napi else {'libc.so'}
    if dependencies != expected:
        raise ValueError(f'Unexpected native dependencies: {sorted(dependencies)}; expected {sorted(expected)}')
    stack = next((line for line in output.splitlines() if 'GNU_STACK' in line), '')
    if not stack or 'E' in stack:
        raise ValueError('Missing or executable GNU_STACK')
    if napi and 'napi_module_register' not in output:
        raise ValueError('Node-API module registration is missing')


def check_c_exports(header, output):
    expected = set(re.findall(r'HYPER_CAPI_EXPORT[^\n]*?\b(HF\w+)\(', header))
    exported = {line.split()[-1] for line in output.splitlines()
                if 'GLOBAL' in line and ' UND ' not in line}
    if len(expected) < 110 or expected - exported:
        raise ValueError(f'Missing C API exports: {sorted(expected - exported)} (found {len(expected)} declarations)')
    print(f'Verified {len(expected)} C API exports', flush=True)


def write_archive(package, destination):
    # Product HAR README is included; local plans, reports and build logs are not.
    for path in package.rglob('*'):
        relative = path.relative_to(package)
        if (path.is_symlink() or any(part.startswith('.') or part == 'prompt_docs' for part in relative.parts)
                or path.suffix in {'.log', '.profraw', '.profdata'}
                or (path.suffix == '.md' and relative.as_posix() != 'HarmonyOS/har/README.md')):
            raise ValueError(f'Unexpected release content: {relative}')
    temporary = destination.with_suffix('.zip.partial')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(package.rglob('*')):
                if path.is_file():
                    archive.write(path, Path(package.name) / path.relative_to(package))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('HarmonyOS ZIP checksum validation failed')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def package_sdk(native_install, napi_install, native_sdk, version, output_dir):
    har = napi_install / 'HarmonyOS/har'
    version = check_version(native_install, har, version)
    for path in (native_install / 'InspireFace/lib/libInspireFace.so',
                 native_install / 'InspireFace/include/inspireface.h',
                 native_install / 'InspireFace/include/inspireface/inspireface.hpp',
                 har / 'Index.ets', har / 'hvigorfile.ts', har / 'build-profile.json5',
                 har / 'src/main/module.json5', har / 'src/main/ets/InspireFace.ets',
                 har / 'src/main/cpp/types/libinspireface_napi/index.d.ts',
                 har / NAPI_LIBRARY):
        if not path.is_file():
            raise ValueError(f'Missing SDK payload: {path}')
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f'inspireface-harmonyos-arm64-v8a-{version}'
    destination = output_dir / f'{name}.zip'
    with tempfile.TemporaryDirectory(prefix='harmony-package-', dir=output_dir) as temporary:
        temporary = Path(temporary)
        package = temporary / name
        ignore = shutil.ignore_patterns('.DS_Store', '._*')
        shutil.copytree(native_install / 'InspireFace/include', package / 'InspireFace/include', ignore=ignore)
        core = package / 'InspireFace/lib/libInspireFace.so'
        core.parent.mkdir(parents=True)
        shutil.copy2(native_install / 'InspireFace/lib/libInspireFace.so', core)
        shutil.copytree(har, package / 'HarmonyOS/har', ignore=ignore)
        napi = package / 'HarmonyOS/har' / NAPI_LIBRARY
        standalone_napi = package / 'HarmonyOS/libs/arm64-v8a/libinspireface_napi.so'
        standalone_napi.parent.mkdir(parents=True)
        shutil.copy2(native_install / 'version.txt', package / 'version.txt')
        for library in (core, napi):
            subprocess.run([str(native_sdk / 'llvm/bin/llvm-strip'), '--strip-unneeded', str(library)], check=True)
            output = subprocess.check_output([str(native_sdk / 'llvm/bin/llvm-readelf'),
                                              '-h', '-d', '-W', '-l', '--dyn-syms', str(library)], text=True)
            check_elf(output, napi=library == napi)
            if library == core:
                check_c_exports((package / 'InspireFace/include/inspireface.h').read_text(), output)
        shutil.copy2(napi, standalone_napi)
        verifier = ROOT / 'cpp/inspireface/platform/ohos/napi'
        subprocess.run(['cmake', f'-DISF_READELF_EXECUTABLE={native_sdk}/llvm/bin/llvm-readelf',
                        f'-DISF_NAPI_LIBRARY={napi}', '-P', str(verifier / 'verify_ohos_napi.cmake')], check=True)
        subprocess.run(['cmake', f'-DISF_NAPI_SOURCE={verifier}/inspireface_napi.cpp',
                        f'-DISF_NAPI_DECLARATION={package}/HarmonyOS/har/src/main/cpp/types/libinspireface_napi/index.d.ts',
                        f'-DISF_ARKTS_WRAPPER={package}/HarmonyOS/har/src/main/ets/InspireFace.ets',
                        f'-DISF_C_API_HEADER={package}/InspireFace/include/inspireface.h',
                        f'-DISF_C_API_PARITY={verifier}/ohos_c_api_parity.cmake',
                        '-P', str(verifier / 'verify_ohos_arkts_contract.cmake')], check=True)
        # Compile and link consumers using only the staged public headers/library.
        # These are cross-compiled executables; no device or simulator is started.
        consumer_build = temporary / 'consumer-build'
        subprocess.run(['cmake', '-S', str(Path(__file__).parent / 'consumer'), '-B', str(consumer_build),
                        f'-DCMAKE_TOOLCHAIN_FILE={native_sdk}/build/cmake/ohos.toolchain.cmake',
                        '-DOHOS_ARCH=arm64-v8a', '-DOHOS_STL=c++_static', '-DCMAKE_BUILD_TYPE=Release',
                        f'-DSDK_DIR={package}/InspireFace'], check=True)
        subprocess.run(['cmake', '--build', str(consumer_build), '--parallel', '2'], check=True)
        write_archive(package, destination)
    checksum = destination.with_suffix('.zip.sha256')
    checksum.write_text(f'{hashlib.sha256(destination.read_bytes()).hexdigest()}  {destination.name}\n')
    print(f'HarmonyOS SDK archive ready: {destination}', flush=True)
    return destination, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-install', type=Path, required=True)
    parser.add_argument('--napi-install', type=Path, required=True)
    parser.add_argument('--native-sdk', type=Path, required=True)
    parser.add_argument('--version', default='')
    parser.add_argument('--output-dir', type=Path, default=Path('build/release'))
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args()
    archive, checksum = package_sdk(args.native_install.resolve(), args.napi_install.resolve(),
                                    args.native_sdk.resolve(), args.version, args.output_dir.resolve())
    if args.github_output:
        with args.github_output.open('a') as output:
            output.write(f'archive={archive}\nchecksum={checksum}\n')


if __name__ == '__main__':
    main()
