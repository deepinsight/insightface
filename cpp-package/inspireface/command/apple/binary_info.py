#!/usr/bin/env python3
"""Inspect Mach-O load commands, including every object inside static archives."""
import re
import subprocess

PLATFORMS = {'1': 'macosx', '2': 'iphoneos', '7': 'iphonesimulator',
             'MACOS': 'macosx', 'IOS': 'iphoneos', 'IOSSIMULATOR': 'iphonesimulator'}


def inspect(binary):
    arches = subprocess.check_output(['xcrun', 'lipo', '-archs', str(binary)], text=True).split()
    result = {}
    for arch in arches:
        text = subprocess.check_output(['xcrun', 'otool', '-l', '-arch', arch, str(binary)], text=True)
        records = []
        for command in re.split(r'Load command \d+\n', text)[1:]:
            if re.search(r'\bcmd LC_BUILD_VERSION\b', command):
                platform = re.search(r'\bplatform (\S+)', command)[1]
                minimum = re.search(r'\bminos (\S+)', command)[1]
                records.append((PLATFORMS.get(platform, platform), minimum))
            elif re.search(r'\bcmd LC_VERSION_MIN_(?:MACOSX|IPHONEOS)\b', command):
                platform = 'macosx' if 'LC_VERSION_MIN_MACOSX' in command else (
                    'iphonesimulator' if arch == 'x86_64' else 'iphoneos')
                records.append((platform, re.search(r'\bversion (\S+)', command)[1]))
        if not records:
            raise RuntimeError(f'No deployment metadata: {binary} ({arch})')
        platforms = {r[0] for r in records}
        if len(platforms) != 1:
            raise RuntimeError(f'Mixed platform archive: {binary} ({arch}): {platforms}')
        maximum = max((r[1] for r in records), key=lambda v: tuple(map(int, v.split('.'))))
        result[arch] = dict(sdk=platforms.pop(), minimum=maximum)
    return result


def validate(binary, sdk, arches):
    info = inspect(binary)
    if set(info) != set(arches) or any(value['sdk'] != sdk for value in info.values()):
        raise RuntimeError(f'Wrong platform/architecture: {binary}: {info}, expected {sdk} {arches}')
    return info
