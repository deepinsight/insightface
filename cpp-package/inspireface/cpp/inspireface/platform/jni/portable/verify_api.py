#!/usr/bin/env python3
"""Verify public C declarations, generated Java methods, and exported JNI symbols agree."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def exports(nm, library):
    symbols = subprocess.check_output([nm, '-g', str(library)], text=True)
    return {fields[-1].lstrip('_') for line in symbols.splitlines()
            if len(fields := line.split()) >= 3 and fields[-2] in ('T', 'W')}


def java_signatures(paths):
    primitives = {'void': 'void', 'boolean': 'jboolean', 'byte': 'jbyte', 'char': 'jchar',
                  'short': 'jshort', 'int': 'jint', 'long': 'jlong', 'float': 'jfloat',
                  'double': 'jdouble', 'String': 'jstring'}

    def jni_type(value):
        if value.endswith('[]'):
            return primitives.get(value[:-2], 'jobject') + 'Array'
        return primitives.get(value, 'jobject')

    def mangle(value):
        return value.replace('_', '_1').replace('.', '_')

    result = {}
    for path in paths:
        source = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(), flags=re.S)
        package = re.search(r'\bpackage\s+([\w.]+)\s*;', source).group(1)
        prefix = 'Java_' + mangle(package + '.' + path.stem) + '_'
        for ret, name, args in re.findall(r'\bnative\s+(\w+(?:\[\])?)\s+(\w+)\s*\((.*?)\)\s*;', source, re.S):
            symbol = prefix + mangle(name)
            if symbol in result:
                raise RuntimeError(f'Overloaded native method needs explicit JNI signature support: {symbol}')
            result[symbol] = [jni_type(ret)] + [jni_type(arg.strip().split()[0])
                                              for arg in args.split(',') if arg.strip()]
    return result


def cpp_signatures(paths):
    result = {}
    for path in paths:
        for ret, name, args in re.findall(
                r'JNIEXPORT\s+(\w+)\s+(?:JNICALL\s+)?(Java_\w+|INSPIRE_FACE_JNI\(\w+\)|JNI_METHOD\(\w+\))\s*\((.*?)\)\s*\{',
                path.read_text(), re.S):
            if name.startswith('INSPIRE_FACE_JNI('):
                name = 'Java_com_insightface_sdk_inspireface_' + name[len('INSPIRE_FACE_JNI('):-1]
            elif name.startswith('JNI_METHOD('):
                name = 'Java_com_insightface_sdk_inspireface_jni_Native_' + name[len('JNI_METHOD('):-1]
            if name in result:
                raise RuntimeError(f'Duplicate JNI implementation: {name}')
            result[name] = [ret] + [re.match(r'\s*(\w+)', arg).group(1) for arg in args.split(',')[2:]]
    return result


def verify_signatures(declarations, implementations, exported):
    for name, signature in declarations.items():
        if implementations.get(name) != signature:
            raise RuntimeError(f'{name}: Java expects {signature}, JNI implements {implementations.get(name)}')
        if name not in exported:
            raise RuntimeError(f'JNI export missing: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--nm', default='nm')
    parser.add_argument('--android-library', type=Path)
    args = parser.parse_args()
    header = (args.root / 'cpp/inspireface/c_api/inspireface.h').read_text()
    expected = set(re.findall(r'HYPER_CAPI_EXPORT\s+extern\s+\w+\s+(HF\w+)\s*\(', header))
    manifest = set(json.loads((args.generated / 'api-manifest.json').read_text()))
    java = (args.generated / 'java/com/insightface/sdk/inspireface/jni/Native.java').read_text()
    methods = set(re.findall(r'public static native \w+ (HF\w+)\(', java))
    test = (args.root / 'java/src/test/java/com/insightface/sdk/inspireface/jni/ContractTest.java').read_text()
    exercised = set(re.findall(r'\b(HF\w+)\(', test)) & expected
    all_exports = exports(args.nm, args.library)
    prefix = 'Java_com_insightface_sdk_inspireface_jni_Native_'
    exported = {symbol[len(prefix):] for symbol in all_exports if symbol.startswith(prefix + 'HF')}
    for name, actual in [('manifest', manifest), ('Java declarations', methods),
                         ('native exports', exported), ('contract call sites', exercised)]:
        if actual != expected:
            raise RuntimeError(f'{name}: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    print(f'Portable JNI API parity: {len(expected)}/{len(expected)} declarations, exports, and contract call sites')
    jni_root = args.root / 'cpp/inspireface/platform/jni'
    declarations = java_signatures(list((args.root / 'java/src/main/java').rglob('*.java')) +
                                   list((args.generated / 'java').rglob('*.java')))
    implementations = cpp_signatures(list((jni_root / 'portable').glob('*.cpp')) + [args.generated / 'bindings.cpp'])
    verify_signatures(declarations, implementations, all_exports)
    if set(declarations) != set(implementations):
        raise RuntimeError(f'JNI implementations without Java declarations: {sorted(set(implementations) - set(declarations))}')
    print(f'Portable JNI signatures and exports: {len(declarations)} (including CPUEngine and ABI helpers)')
    if args.android_library:
        legacy = json.loads((jni_root / 'android/legacy-api.json').read_text())['signatures']
        supplemental = java_signatures((jni_root / 'java').rglob('*.java'))
        declarations = dict(legacy, **supplemental)
        implementations = cpp_signatures((jni_root / 'android').glob('*.cpp'))
        verify_signatures(declarations, implementations, exports(args.nm, args.android_library))
        print(f'Android JNI compatibility: {len(legacy)} legacy + {len(supplemental)} supplemental Java declarations')


if __name__ == '__main__':
    main()
