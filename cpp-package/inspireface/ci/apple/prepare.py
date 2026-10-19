#!/usr/bin/env python3
"""Prepare deterministic Apple CI inputs and a cache key without publishing files."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def main():
    version = os.environ.get('VERSION', '').removeprefix('v')
    if version and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+_-]*', version):
        raise ValueError('Invalid release tag/version')
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write(f'VERSION={version}\n')
    dependency = ROOT / '3rdparty'
    if not dependency.exists():
        subprocess.run(['git', 'clone', '--recurse-submodules',
                        'https://github.com/tunmx/inspireface-3rdparty.git', str(dependency)], check=True)
    if os.environ.get('ISF_APPLE_TEST_FIXTURES') == 'true':
        model = ROOT / 'test_res/pack/Pikachu'
        if not model.is_file():
            subprocess.run(['bash', str(ROOT / 'command/download_models_general.sh'), 'Pikachu'], cwd=ROOT, check=True)
        if not (ROOT / 'test_res/data/bulk/kun.jpg').is_file():
            raise RuntimeError('Missing tracked Apple model test image')
    identity = dict(toolchain=output('xcodebuild', '-version'), compiler=output('xcrun', 'clang', '--version'),
                    cmake=output('cmake', '--version'), architecture=output('uname', '-m'),
                    macos=os.environ.get('MACOSX_DEPLOYMENT_TARGET', ''),
                    ios=os.environ.get('IOS_DEPLOYMENT_TARGET', '11.0'),
                    dependencies=output('git', '-C', str(dependency), 'submodule', 'status', '--recursive'),
                    root=output('git', '-C', str(dependency), 'rev-parse', 'HEAD'),
                    sdks={sdk: output('xcrun', '--sdk', sdk, '--show-sdk-version')
                          for sdk in ('macosx', 'iphoneos', 'iphonesimulator')})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    print(f'Apple toolchain/dependency cache: {key}')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write(f'cache-key={key}\n')


if __name__ == '__main__':
    main()
