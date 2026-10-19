#!/usr/bin/env python3
"""Install only the Linux native toolchain from the pinned OpenHarmony public SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


SDK_VERSION = '6.1.0.31'
SDK_URL = ('https://repo.huaweicloud.com/openharmony/os/6.1-Release/'
           'ohos-sdk-windows_linux-public.tar.gz')
# Published alongside the official 6.1 Release archive as .tar.gz.sha256.
SDK_SHA256 = 'b833b75a64ee46bbd7880921abbb49b733ec5c8171b6684c9b524d57f624cee0'
NATIVE_ARCHIVE = f'native-linux-x64-{SDK_VERSION}-Release.zip'


def check_sdk(native):
    metadata = json.loads((native / 'oh-uni-package.json').read_text())
    if metadata.get('version') != SDK_VERSION or metadata.get('apiVersion') != '23':
        raise ValueError(f'Unexpected OpenHarmony SDK metadata: {metadata}')
    for relative in ('build/cmake/ohos.toolchain.cmake', 'llvm/bin/clang',
                     'llvm/bin/clang++', 'llvm/bin/llvm-readelf', 'llvm/bin/llvm-strip',
                     'build-tools/cmake/bin/cmake',
                     'sysroot/usr/include/napi/native_api.h'):
        if not (native / relative).is_file():
            raise ValueError(f'Incomplete OpenHarmony SDK: {relative}')


def install_sdk(output, archive=None):
    if output.exists():
        check_sdk(output)
        print(f'Using cached OpenHarmony {SDK_VERSION}: {output}', flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ohos-sdk-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        if archive is None:
            archive = temporary / 'sdk.tar.gz'
            print(f'Downloading OpenHarmony {SDK_VERSION} public SDK', flush=True)
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--retry', '3', '--connect-timeout', '30', '--max-time', '1800',
                            '--output', str(archive), SDK_URL], check=True)
        digest = hashlib.sha256()
        with archive.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != SDK_SHA256:
            raise ValueError('OpenHarmony SDK archive SHA256 does not match the pinned release')
        native_zip = temporary / NATIVE_ARCHIVE
        with tarfile.open(archive, 'r:gz') as bundle:
            matches = [member for member in bundle.getmembers()
                       if member.isfile() and Path(member.name).name == NATIVE_ARCHIVE]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {NATIVE_ARCHIVE} in the public SDK')
            with bundle.extractfile(matches[0]) as source, native_zip.open('wb') as target:
                shutil.copyfileobj(source, target)
        # unzip preserves the executable modes and LLVM symlinks in the official ZIP.
        extracted = temporary / 'extracted'
        subprocess.run(['unzip', '-oq', str(native_zip), '-d', str(extracted)], check=True)
        native = extracted / 'native'
        check_sdk(native)
        shutil.move(str(native), str(output))
    print(f'Installed OpenHarmony {SDK_VERSION}: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official archive')
    args = parser.parse_args()
    install_sdk(args.output.resolve(), args.archive)


if __name__ == '__main__':
    main()
