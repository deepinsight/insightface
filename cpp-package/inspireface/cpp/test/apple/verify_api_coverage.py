#!/usr/bin/env python3
"""Require explicit Apple mappings; optionally verify each method ran in BOTH languages."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
EXCLUDED = {
    'HFQueryExpansiveHardwareRGACompileOption': 'Rockchip RGA',
    'HFSetExpansiveHardwareRockchipDmaHeapPath': 'Rockchip DMA',
    'HFQueryExpansiveHardwareRockchipDmaHeapPath': 'Rockchip DMA',
    'HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize': 'Rockchip DMA',
    'HFSetCudaDeviceId': 'CUDA', 'HFGetCudaDeviceId': 'CUDA', 'HFPrintCudaDeviceInfo': 'CUDA',
    'HFGetNumCudaDevices': 'CUDA', 'HFCheckCudaDeviceSupport': 'CUDA',
}


def mappings():
    header = (ROOT / 'cpp/inspireface/platform/apple/include/IFInspireFace.h').read_text()
    result = {}
    for match in re.finditer(r'@interface\s+(\w+)\s*:[\s\S]*?@end', header):
        owner = match[1]
        for entry in re.finditer(r'// C API: (HF\w+)\s+([+-])\s*\([^)]*\)([^;]+);', match[0]):
            symbol, kind, declaration = entry.groups()
            declaration = declaration.split('NS_SWIFT_NAME')[0]
            labels = re.findall(r'(\w+)\s*:', declaration)
            selector = ''.join(label + ':' for label in labels) if labels else declaration.strip()
            if symbol in result:
                raise RuntimeError(f'Duplicate mapping for {symbol}')
            result[symbol] = f'{kind}[{owner} {selector}]'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=Path)
    parser.add_argument('--profiles', type=Path)
    args = parser.parse_args()
    api = set(re.findall(r'HYPER_CAPI_EXPORT\s+extern\s+\w+\s+(HF\w+)\s*\(', (ROOT / 'cpp/inspireface/c_api/inspireface.h').read_text()))
    mapping = mappings()
    expected = api - EXCLUDED.keys()
    if set(mapping) != expected or not EXCLUDED.keys() <= api:
        raise RuntimeError(f'Unmapped: {sorted(expected-set(mapping))}; stale: {sorted(set(mapping)-expected)}')
    implementation = (ROOT / 'cpp/inspireface/platform/apple/IFInspireFace.m').read_text()
    for symbol in expected:
        if not re.search(r'\b' + symbol + r'\s*\(', implementation):
            raise RuntimeError(f'Missing C delegation: {symbol}')
    print(f'Apple API mapping: {len(mapping)}/{len(expected)}; {len(EXCLUDED)} explicit hardware exclusions')
    if args.framework or args.profiles:
        if not (args.framework and args.profiles):
            parser.error('--framework and --profiles must be supplied together')
        for language in ('objc', 'swift'):
            directory = args.profiles / language
            raw = sorted(directory.glob('*.profraw'))
            if not raw:
                raise RuntimeError(f'No execution profiles for {language}')
            merged = directory / 'coverage.profdata'
            subprocess.run(['xcrun', 'llvm-profdata', 'merge', '-sparse', *map(str, raw), '-o', str(merged)], check=True)
            output = subprocess.check_output(['xcrun', 'llvm-cov', 'export', str(args.framework), '-instr-profile', str(merged)], text=True)
            data = json.loads(output)
            executed = {match[0] for record in data['data'] for f in record.get('functions', [])
                        if f['count'] > 0 for match in [re.search(r'[+-]\[.*\]$', f['name'])] if match}
            missing = {symbol: method for symbol, method in mapping.items() if method not in executed}
            if missing:
                raise RuntimeError(f'{language} did not execute these adapters: {missing}')
            print(f'{language}: {len(mapping)} mapped methods executed against the real SDK')


if __name__ == '__main__':
    main()
