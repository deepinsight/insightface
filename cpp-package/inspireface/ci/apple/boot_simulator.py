#!/usr/bin/env python3
"""Boot an available iOS simulator for this runner; fail when none is available."""
import json
import os
import subprocess


def main():
    devices = json.loads(subprocess.check_output(['xcrun', 'simctl', 'list', 'devices', 'available', '--json']))['devices']
    candidates = [(runtime, device) for runtime, values in devices.items() if '.iOS-' in runtime
                  for device in values if device.get('isAvailable') and device['name'].startswith('iPhone')]
    if not candidates:
        raise RuntimeError('No installed iOS simulator. Select an Xcode image with an iOS runtime.')
    # Prefer the newest installed runtime. Avoid downloading another multi-GB runtime in CI.
    candidates.sort(key=lambda item: tuple(map(int, item[0].split('.iOS-')[1].split('-'))), reverse=True)
    device = candidates[0][1]
    if device['state'] != 'Booted':
        subprocess.run(['xcrun', 'simctl', 'boot', device['udid']], check=True)
    subprocess.run(['xcrun', 'simctl', 'bootstatus', device['udid'], '-b'], check=True, timeout=240)
    print(f'Simulator ready: {device["name"]} ({device["udid"]})')
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write(f'ISF_APPLE_SIMULATOR={device["udid"]}\n')


if __name__ == '__main__':
    main()
