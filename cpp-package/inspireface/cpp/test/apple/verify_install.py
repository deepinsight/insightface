#!/usr/bin/env python3
"""Compile installed C/C++/Objective-C/Swift consumers for an Apple slice."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'command/apple'))
from binary_info import inspect, validate

p = argparse.ArgumentParser()
p.add_argument('--sdk', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--arch', choices=['arm64', 'x86_64'])
p.add_argument('--run-simulator', action='store_true')
p.add_argument('--compile-only', action='store_true', help='Validate a transferred slice without executing another architecture')
a = p.parse_args()
a.sdk = a.sdk.resolve()
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
for entry in a.sdk.rglob('*'):
    if entry.is_symlink() and a.sdk not in entry.resolve(strict=True).parents:
        raise RuntimeError(f'SDK symlink escapes the package: {entry}')
source = Path(__file__).resolve().parent
core = a.sdk / 'InspireFace.framework/InspireFace'
metadata = inspect(core)
arch = a.arch or (platform.machine() if platform.machine() in metadata else next(iter(metadata)))
sdk_name, minimum = metadata[arch]['sdk'], metadata[arch]['minimum']
overlay_metadata = validate(a.sdk / 'InspireFaceSwift.framework/InspireFaceSwift', sdk_name, list(metadata))
if overlay_metadata[arch]['minimum'] != minimum:
    raise RuntimeError(f'Core and Swift deployment targets differ: {metadata}, {overlay_metadata}')
sdk_path = subprocess.check_output(['xcrun', '--sdk', sdk_name, '--show-sdk-path'], text=True).strip()
os_name = 'macosx' if sdk_name == 'macosx' else 'ios'
suffix = '-simulator' if sdk_name == 'iphonesimulator' else ''
target = f'{arch}-apple-{os_name}{minimum}{suffix}'
compile_args = ['-target', target, '-isysroot', sdk_path]
legacy = a.sdk / 'InspireFace'
assert (legacy / 'include/inspireface.h').is_file()
env = dict(os.environ, LLVM_PROFILE_FILE=str(a.output / 'consumer-%p-%m.profraw'))
system_links = ['-framework', 'Foundation', '-framework', 'CoreVideo', '-framework', 'CoreML', '-framework', 'Accelerate', '-lc++']
framework_base = ['-F', str(a.sdk), '-framework', 'InspireFace', '-Wl,-rpath,' + str(a.sdk)]
frameworks = [*framework_base, *system_links]
raw = legacy / 'lib/libInspireFace.dylib'
if raw.exists():
    raw_links = [str(raw), '-Wl,-rpath,' + str(raw.parent)]
else:
    raw = legacy / 'lib/libInspireFace.a'
    validate(legacy / 'lib/libMNN.a', sdk_name, [arch])
    raw_links = [str(raw), str(legacy / 'lib/libMNN.a'), *system_links]
validate(raw, sdk_name, [arch])
if sdk_name == 'macosx':
    for binary in [core, a.sdk / 'InspireFaceSwift.framework/InspireFaceSwift'] + ([raw] if raw.suffix == '.dylib' else []):
        dependencies = subprocess.check_output(['xcrun', 'otool', '-L', str(binary)], text=True)
        for line in dependencies.splitlines():
            if line.startswith('\t') and not line.strip().startswith(('@rpath/', '/System/Library/', '/usr/lib/')):
                raise RuntimeError(f'Nonportable dependency in {binary}: {line}')


def execute(binary):
    if a.compile_only:
        return
    if sdk_name == 'macosx':
        subprocess.run(['arch', '-' + arch, str(binary)], env=env, check=True)
    elif sdk_name == 'iphonesimulator' and a.run_simulator:
        subprocess.run([sys.executable, str(source / 'run_simulator.py'), str(binary)], env=env, check=True)


consumers = [('raw', raw_links), ('framework', frameworks)]
if sdk_name != 'macosx':
    validate(a.sdk / 'MNN.framework/MNN', sdk_name, [arch])
    consumers.append(('raw-framework-dependency', [str(raw), '-F', str(a.sdk), '-framework', 'MNN', *system_links]))
for kind, links in consumers:
    object_file = a.output / f'{kind}-c.o'
    binary = a.output / f'{kind}-c'
    includes = ['-I', str(legacy / 'include')]
    if kind == 'framework':
        includes = ['-F', str(a.sdk), '-DIF_TEST_FRAMEWORK=1']
    subprocess.run(['xcrun', 'clang', *compile_args, *includes, '-c', str(source / 'legacy_consumer.c'), '-o', str(object_file)], check=True)
    subprocess.run(['xcrun', 'clang++', *compile_args, str(object_file), *links, '-o', str(binary)], check=True)
    execute(binary)
    binary = a.output / f'{kind}-cpp'
    subprocess.run(['xcrun', 'clang++', *compile_args, '-std=c++14', '-I', str(legacy / 'include'),
                    str(source / 'legacy_consumer.cpp'), *links, '-o', str(binary)], check=True)
    execute(binary)

objc = a.output / 'consumer.m'
objc.write_text('@import InspireFace;\nint main() { @autoreleasepool { NSError *error = nil; IFImageStream *s = [[IFImageStream alloc] initEmptyWithError:&error]; if (!s || error) return 1; return [s closeWithError:&error] ? 0 : 2; } }\n')
binary = a.output / 'objc-consumer'
subprocess.run(['xcrun', 'clang', *compile_args, '-fobjc-arc', '-fmodules',
                '-fmodules-cache-path=' + str(a.output / '.clang-module-cache'), str(objc),
                *framework_base, '-Wl,-ObjC', '-o', str(binary)], check=True)
execute(binary)
consumer = a.output / 'consumer.swift'
consumer.write_text('import InspireFaceSwift\nvar level: UInt32 = 0\ntry InspireFaceDiagnostics.getCAPILevel(&level)\nprecondition(level == HF_C_API_LEVEL)\nlet stream = try ImageStream()\ntry stream.close()\n')
swift_args = ['xcrun', '--sdk', sdk_name, 'swiftc', '-target', target, '-sdk', sdk_path,
              '-module-cache-path', str(a.output / '.module-cache')]
binary = a.output / 'swift-consumer'
subprocess.run([*swift_args, '-F', str(a.sdk), '-framework', 'InspireFace', '-framework', 'InspireFaceSwift',
                '-Xlinker', '-ObjC', '-Xlinker', '-rpath', '-Xlinker', str(a.sdk),
                str(consumer), '-o', str(binary)], check=True)
execute(binary)
# Relocate the frameworks, remove serialized modules, and exercise public Swift interfaces.
with tempfile.TemporaryDirectory(prefix='interface-', dir=a.output) as temp:
    temp = Path(temp)
    for name in ('InspireFace.framework', 'InspireFaceSwift.framework'):
        shutil.copytree(a.sdk / name, temp / name, symlinks=True)
    for file in (temp / 'InspireFaceSwift.framework').rglob('*.swiftmodule'):
        if file.is_file():
            file.unlink()
    subprocess.run([*swift_args, '-F', str(temp), '-typecheck', str(consumer)], check=True)
result = dict(platform=sdk_name, arch=arch, deployment=minimum, consumers='C/C++/Objective-C/Swift',
              compiled=True, executed=not a.compile_only and (sdk_name == 'macosx' or a.run_simulator), interfaceOnlyImport=True)
(a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('Installed SDK validation:', json.dumps(result))
