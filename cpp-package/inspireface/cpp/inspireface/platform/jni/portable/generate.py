#!/usr/bin/env python3
"""Generate the portable Java/JNI C API binding. Unknown shapes fail the build."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PACKAGE = 'com.insightface.sdk.inspireface.jni'
PREFIX = 'Java_com_insightface_sdk_inspireface_jni_Native_'
SCALARS = {
    'HInt32': ('int', 'Int', 'I'), 'HOption': ('int', 'Int', 'I'),
    'HFUInt32': ('int', 'Int', 'I'), 'HFUInt64': ('long', 'Long', 'J'),
    'HFaceId': ('long', 'Long', 'J'), 'HFloat': ('float', 'Float', 'F'),
    'HChar': ('byte', 'Byte', 'B'), 'HResult': ('long', 'Long', 'J'),
    'HFStatus': ('int', 'Int', 'I'),
}
HANDLES = {'HFSession', 'HFImageStream', 'HFImageBitmap', 'HFFaceResultSnapshot', 'HFFaceCaptureSession'}
POINTERS = {'HPFloat': 'HFloat', 'HPInt32': 'HInt32', 'HPFaceId': 'HFaceId',
            'HPUInt8': 'uint8_t', 'HPVoid': 'uint8_t'}
# These pointer fields borrow native memory, rather than copying pixels/features/results.
COUNTS = {
    'HFImageData.data': 'ImageBytes(env, v.format, v.width, v.height)',
    'HFImageBitmapData.data': 'BitmapBytes(env, v.width, v.height, v.channels)',
    'HFFaceBasicToken.data': 'v.size', 'HFFaceFeature.data': 'v.size',
    'HFSearchTopKResults.confidence': 'v.size', 'HFSearchTopKResults.ids': 'v.size',
    'HFFeatureHubExistingIds.ids': 'v.size',
}
STRUCT_ARRAYS = {'HFMultipleFaceData.rects': ('HFaceRect', 'v.detectedNum'),
                 'HFMultipleFaceData.tokens': ('HFFaceBasicToken', 'v.detectedNum')}
IN_STRUCT_POINTERS = {
    ('HFCreateImageStream', 'data'), ('HFCreateImageBitmap', 'data'),
    ('HFCreateInspireFaceSessionV2', 'config'), ('HFCreateFaceCaptureSession', 'config'),
    ('HFMultipleFacePipelineProcess', 'faces'), ('HFMultipleFacePipelineProcessOptional', 'faces'),
    ('HFReleaseFaceFeature', 'feature'),
}
ARRAY_ARGS = {'HFGetFaceCaptureResults': ('results', 'HFFaceCaptureResult', 'capacity'),
              'HFGetFaceDenseLandmarkFromFaceToken': ('landmarks', 'HPoint2f', 'num'),
              'HFGetFaceFiveKeyPointsFromFaceToken': ('landmarks', 'HPoint2f', 'num')}
VERSIONS = {'HFSessionConfigV2': 'HF_SESSION_CONFIG_V2_VERSION',
            'HFResourcePackInfo': 'HF_RESOURCE_PACK_INFO_VERSION',
            'HFFaceCaptureConfig': 'HF_FACE_CAPTURE_CONFIG_VERSION'}
CUSTOM = {'HFCreateImageStream', 'HFCreateImageStreamEmpty', 'HFImageStreamSetBuffer',
          'HFImageStreamSetFormat', 'HFReleaseImageStream', 'HFCreateImageStreamFromImageBitmap',
          'HFCreateFaceFeature', 'HFReleaseFaceFeature'}


def clean(text):
    return re.sub(r'//[^\n]*', '', re.sub(r'/\*.*?\*/', '', text, flags=re.S))


def declaration(text):
    match = re.fullmatch(r'\s*(.*?)\s*(\w+)\s*(?:\[([^\]]+)\])?\s*', text)
    if not match:
        raise ValueError('Unsupported declaration: ' + text)
    kind, name, length = match.groups()
    return re.sub(r'\s*\*\s*', '*', kind.strip()), name, length


def generate(root, output):
    capi = clean((root / 'cpp/inspireface/c_api/inspireface.h').read_text())
    types = clean((root / 'cpp/inspireface/c_api/intypedef.h').read_text()) + capi
    errors = clean((root / 'cpp/inspireface/include/inspireface/herror.h').read_text())
    structs = {}
    enums = {}
    constants = {}
    for name, body in re.findall(r'typedef\s+enum\s+(\w+)\s*\{([^}]+)\}', types):
        SCALARS[name] = ('int', 'Int', 'I')
        value = -1
        for item in body.split(','):
            if not item.strip():
                continue
            parts = item.strip().split('=')
            value = int(parts[1].strip(), 0) if len(parts) == 2 else value + 1
            enums[parts[0].strip()] = value
    constants.update(enums)
    for name, expression in re.findall(r'^\s*#define\s+((?:HF_|HERR_|HSUCCEED)\w*)[ \t]+([^\n]+)', capi + '\n' + errors, re.M):
        expression = re.sub(r'UINT64_C\((\d+)\)', r'\1', expression)
        expression = re.sub(r'\((?:HFaceId|HFUInt64|HFUInt32)\)', '', expression)
        expression = re.sub(r'(?<=\d)[uUlL]+\b', '', expression.strip())
        for token in re.findall(r'\b(?:HF_|HERR_)\w+', expression):
            expression = expression.replace(token, str(constants[token]))
        if not re.fullmatch(r'[\dxa-fA-F()+<>\s|&~-]+', expression):
            raise ValueError('Unknown constant expression: ' + expression)
        constants[name] = eval(expression, {'__builtins__': {}}, {})
    for name, body in re.findall(r'typedef\s+struct\s+(\w+)\s*\{([^}]+)\}', types):
        structs[name] = [declaration(x) for x in body.split(';') if x.strip()]
    functions = []
    for ret, name, args in re.findall(r'HYPER_CAPI_EXPORT\s+extern\s+(\w+)\s+(HF\w+)\s*\((.*?)\)\s*;', capi, re.S):
        functions.append((ret, name, [declaration(x) if x.strip() != '...' else ('...', '', None)
                                      for x in args.split(',') if x.strip()]))
    if len({n for _, n, _ in functions}) != len(functions):
        raise ValueError('Duplicate C API function')

    def count_for(owner, field):
        if owner == 'HFMultipleFaceData':
            return 'v.detectedNum'
        if owner == 'HFFaceEulerAngle':
            return 'count'
        return COUNTS.get(owner + '.' + field, 'v.num' if any(n == 'num' for _, n, _ in structs[owner]) else None)

    def field_type(owner, kind, name, length):
        if length:
            j, _, sig = SCALARS[kind]
            return j + '[]', '[' + sig
        if kind in SCALARS:
            j, _, sig = SCALARS[kind]
            return j, sig
        if kind in POINTERS:
            if count_for(owner, name) is None:
                raise ValueError('Pointer requires explicit bounds: ' + owner + '.' + name)
            return 'ByteBuffer', 'Ljava/nio/ByteBuffer;'
        if kind == 'HString':
            return 'String', 'Ljava/lang/String;'
        if owner + '.' + name in STRUCT_ARRAYS:
            t = STRUCT_ARRAYS[owner + '.' + name][0]
            return t + '[]', '[L' + PACKAGE.replace('.', '/') + '/NativeTypes$' + t + ';'
        t = kind[1:] if kind.startswith('P') else kind
        if t in structs:
            return t, 'L' + PACKAGE.replace('.', '/') + '/NativeTypes$' + t + ';'
        raise ValueError('Unsupported field: ' + kind)

    java_types = [f'package {PACKAGE};', 'import java.nio.ByteBuffer;',
                  '/** C descriptors. Native buffer views follow the C API lifetime; see Native for the lifetime contract. */',
                  'public final class NativeTypes {', '  private NativeTypes() {}']
    cpp = ['// Generated from the public C headers. Do not edit.', '#include "support.h"', 'using namespace isf_jni;']
    for t in structs:
        cpp += [f'static void Read(JNIEnv*, jobject, {t}&, Context&, int = 0);',
                f'static jobject Write(JNIEnv*, const {t}&, int = 0);',
                f'static void WriteInto(JNIEnv*, jobject, const {t}&, int = 0);']
    for type_id, (owner, fields) in enumerate(structs.items()):
        java_types += [f'  public static final class {owner}{" implements AutoCloseable" if owner == "HFFaceFeature" else ""} {{']
        if owner == 'HFFaceFeature':
            java_types += ['    private long nativeOwner;',
                           '    /** Releases only HFCreateFaceFeature storage; borrowed features must not be closed. */',
                           '    @Override public void close() { InspireFaceException.check(Native.HFReleaseFaceFeature(this)); }']
        read = [f'static void Read(JNIEnv* env, jobject object, {owner}& v, Context& context, int count) {{',
                '  Require(env, object != nullptr, "Null input descriptor");']
        # Bounds depend on scalar fields, which must be decoded first.
        ordered = sorted(fields, key=lambda f: 0 if f[0] in SCALARS and not f[2] else 1)
        write = [f'static void WriteInto(JNIEnv* env, jobject object, const {owner}& v, int count) {{',
                 '  Require(env, object != nullptr, "Null output descriptor");']
        if owner == 'HFFaceFeature':
            write += ['  auto owner = env->GetLongField(object, Field(env, object, "nativeOwner", "J")); Check(env);',
                      '  Require(env, owner == 0, "Cannot overwrite an owned feature with a borrowed result");']
        for kind, name, length in ordered:
            jtype, sig = field_type(owner, kind, name, length)
            initial = ''
            if length:
                length = constants.get(length, length)
                initial = f' = new {jtype[:-2]}[{length}]'
            elif kind in structs:
                initial = f' = new {kind}()'
            elif name == 'structSize':
                initial = f' = Native.structSize({type_id})'
            elif name == 'structVersion':
                initial = f' = {constants[VERSIONS[owner]]}'
            java_types.append(f'    public {jtype} {name}{initial};')
            field = f'Field(env, object, "{name}", "{sig}")'
            if kind in SCALARS and not length:
                _, jni, _ = SCALARS[kind]
                read += [f'  v.{name} = static_cast<{kind}>(env->Get{jni}Field(object, {field})); Check(env);']
                write += [f'  env->Set{jni}Field(object, {field}, static_cast<j{SCALARS[kind][0]}>(v.{name})); Check(env);']
                continue
            read += [f'  {{ Local value(env, env->GetObjectField(object, {field})); Check(env);']
            if length:
                jni = SCALARS[kind][1]
                read += [f'    ReadArray<{kind}>(env, value.get(), v.{name}, {length});']
                expression = f'WriteArray(env, v.{name}, {length})'
            elif kind in POINTERS:
                n = count_for(owner, name)
                ctype = POINTERS[kind]
                read += [f'    v.{name} = Direct<{ctype}>(env, value.get(), {n});']
                expression = f'Buffer(env, v.{name}, Bytes(env, {n}, sizeof({ctype})))'
            elif kind == 'HString':
                read += [f'    v.{name} = context.string(env, static_cast<jstring>(value.get()));']
                # Only persistenceDbPath uses HString, and it is an input field.
                expression = f'Utf8String(env, v.{name})'
            elif owner + '.' + name in STRUCT_ARRAYS:
                t, n = STRUCT_ARRAYS[owner + '.' + name]
                read += [f'    v.{name} = context.array<{t}>(env, {n});',
                         f'    RequireArray(env, value.get(), {n});',
                         f'    for (int i = 0; i < {n}; ++i) {{',
                         '      Local element(env, env->GetObjectArrayElement(static_cast<jobjectArray>(value.get()), i)); Check(env);',
                         f'      Read(env, element.get(), v.{name}[i], context);', '    }']
                expression = f'WriteStructArray(env, v.{name}, {n}, "{t}", [](JNIEnv* e, const {t}& x) {{ return Write(e, x); }})'
            else:
                t = kind[1:] if kind.startswith('P') else kind
                n = ', v.detectedNum' if t == 'HFFaceEulerAngle' else ''
                if kind.startswith('P'):
                    read += [f'    if (value.get()) {{ v.{name} = context.array<{t}>(env, 1); Read(env, value.get(), *v.{name}, context); }}']
                    expression = f'(v.{name} ? Write(env, *v.{name}) : nullptr)'
                else:
                    read += [f'    Read(env, value.get(), v.{name}, context{n});']
                    expression = f'Write(env, v.{name}{n})'
            read += ['  }']
            write += [f'  {{ Local value(env, {expression});',
                      f'    env->SetObjectField(object, {field}, value.get()); Check(env); }}']
        java_types += ['  }']
        cpp += read + ['}'] + write + ['}']
        cpp += [f'static jobject Write(JNIEnv* env, const {owner}& v, int count) {{',
                f'  Local object(env, NewDescriptor(env, "{owner}"));',
                '  WriteInto(env, object.get(), v, count); return object.release();', '}']
    java_types += ['}']
    abi = hashlib.sha256(json.dumps([structs, functions, constants], sort_keys=True).encode()).hexdigest()
    java = [f'package {PACKAGE};', 'import java.nio.ByteBuffer;',
            f'import {PACKAGE}.NativeTypes.*;', '/** Complete low-level C API. Status codes and borrowed-buffer lifetimes are preserved.',
            ' * Buffers must be writable and direct. Position/limit are honored; use native byte order.',
            ' * Input image buffers are retained until stream release or buffer replacement.',
            ' * Borrowed output buffers do not own native storage: retain the bitmap/session/snapshot',
            ' * and read them before the next operation that invalidates the corresponding C result.',
            ' * Face metadata arrays are marshalled; pixel, token and feature buffers are not copied.',
            ' * Never release a borrowed feature with HFReleaseFaceFeature; use that only for HFCreateFaceFeature.',
            ' * Unsigned integers are represented by the same bits in Java int/long.',
            ' * HFLogPrint accepts literal text. Format arguments in Java before calling.',
            ' * Serialize use of each session and process-wide runtime/FeatureHub state.',
            ' * Caller-owned native resources must be explicitly released.',
            ' */', 'public final class Native {',
            f'  static {{ NativeLibrary.load(); verifyAbi("{abi}"); }}', '  private Native() {}',
            '  private static native void verifyAbi(String expected);',
            '  static native int structSize(int type);']
    cpp += [f'extern "C" JNIEXPORT void JNICALL {PREFIX}verifyAbi(JNIEnv* env, jclass, jstring expected) {{',
            '  try { Context context; auto value = context.string(env, expected);',
            f'    if (!value || std::strcmp(value, "{abi}") != 0)',
            '      Fail(env, "java/lang/UnsatisfiedLinkError", "InspireFace Java/JNI ABI mismatch; use the matching JAR and native library");',
            '  } catch (...) { TranslateException(env); }', '}']
    cpp += [f'extern "C" JNIEXPORT jint JNICALL {PREFIX}structSize(JNIEnv* env, jclass, jint type) {{',
            '  switch (type) {']
    cpp += [f'    case {i}: return sizeof({t});' for i, t in enumerate(structs)]
    cpp += ['    default: return 0;', '  }', '}']
    manifest = {}
    for ret, name, args in functions:
        jret = 'void' if ret == 'void' else SCALARS[ret][0]
        jargs, cargs, pre, post, call = [], [], [], [], []
        for kind, arg, _ in args:
            if kind == '...':
                continue
            local = 'n_' + arg
            ctype = kind.replace('const ', '')
            base = ctype[:-1] if ctype.endswith('*') else ctype[1:] if ctype.startswith('P') else None
            if name in ARRAY_ARGS and ARRAY_ARGS[name][0] == arg:
                _, t, n = ARRAY_ARGS[name]
                jt, ct = t + '[]', 'jobjectArray'
                pre += [f'RequireArray(env, {arg}, {n});', f'auto {local} = context.array<{t}>(env, {n});']
                length = '*n_resultCount' if name == 'HFGetFaceCaptureResults' else n
                post += [f'if (call_status == HSUCCEED && {local}) FillStructArray(env, {arg}, {local}, {length}, [](JNIEnv* e, const {t}& x) {{ return Write(e, x); }});']
                call.append(local)
            elif kind in HANDLES:
                jt, ct = 'long', 'jlong'
                call.append(f'reinterpret_cast<{kind}>(static_cast<uintptr_t>({arg}))')
            elif kind in SCALARS:
                jt, ct = SCALARS[kind][0], 'j' + SCALARS[kind][0]
                call.append(f'static_cast<{kind}>({arg})')
            elif kind in ('HPath', 'HFormat') or (kind == 'HString' and arg == 'title'):
                jt, ct = 'String', 'jstring'
                pre.append(f'auto {local} = context.string(env, {arg});')
                call.append(local)
            elif kind in structs or base in structs:
                t = kind if kind in structs else base
                jt, ct = t, 'jobject'
                pre.append(f'{t} {local}{{}};')
                is_input = kind in structs or (name, arg) in IN_STRUCT_POINTERS
                if is_input:
                    pre.append(f'Read(env, {arg}, {local}, context);')
                else:
                    if name != 'HFValidateResourcePack':
                        pre.append(f'Require(env, {arg} != nullptr, "Null output descriptor: {arg}");')
                    if t in VERSIONS:
                        if name == 'HFValidateResourcePack':
                            pre += [f'if ({arg}) {{',
                                    f'  {local}.structSize = env->GetIntField({arg}, Field(env, {arg}, "structSize", "I")); Check(env);',
                                    f'  {local}.structVersion = env->GetIntField({arg}, Field(env, {arg}, "structVersion", "I")); Check(env);',
                                    '}']
                        else:
                            pre.append(f'{local}.structSize = sizeof({t}); {local}.structVersion = {VERSIONS[t]};')
                call.append(local if kind in structs else f'({arg} ? &{local} : nullptr)')
                if not is_input or name == 'HFReleaseFaceFeature':
                    post.append(f'if (call_status == HSUCCEED && {arg}) WriteInto(env, {arg}, {local});')
            elif kind == 'HPUInt8' or kind == 'HPBuffer' or kind == 'HString' or (name == 'HFFaceFeatureExtractCpy' and arg == 'feature'):
                jt, ct = 'ByteBuffer', 'jobject'
                if kind == 'HPUInt8':
                    # The custom stream implementation knows the current format.
                    n, element = '0', 'uint8_t'
                elif kind == 'HPBuffer':
                    n, element = 'bufferSize', 'char'
                elif name == 'HFFaceFeatureExtractCpy':
                    pre += ['HInt32 featureLength = 0;',
                            'HResult featureStatus = HFGetFeatureLength(&featureLength);',
                            'if (featureStatus != HSUCCEED) return featureStatus;']
                    n, element = 'featureLength', 'float'
                else:
                    n, element = ('256' if name == 'HFQueryExpansiveHardwareRockchipDmaHeapPath' else 'bufferSize'), 'char'
                pre.append(f'auto {local} = Direct<{element}>(env, {arg}, {n});')
                call.append(local)
            elif base in HANDLES or kind in ('HPInt32', 'HPFaceId', 'HPFloat', 'HFUInt32*'):
                t = base if base in HANDLES else {'HPInt32': 'HInt32', 'HPFaceId': 'HFaceId', 'HPFloat': 'HFloat', 'HFUInt32*': 'HFUInt32'}[kind]
                jt = 'long' if base in HANDLES else SCALARS[t][0]
                ct = 'j' + jt + 'Array'
                jt += '[]'
                n = 'count' if name in ('HFDeBugGetUnreleasedSessions', 'HFDeBugGetUnreleasedStreams') else '1'
                pre += [f'RequireArray(env, {arg}, {n});', f'auto {local} = context.array<{t}>(env, {n});']
                post += [f'SetArray(env, {arg}, {local}, {n});']
                call.append(local)
            else:
                raise ValueError(f'Unmapped argument: {name} {kind} {arg}')
            jargs.append(jt + ' ' + arg)
            cargs.append(ct + ' ' + arg)
        if name == 'HFLogPrint':
            pre.append('Require(env, n_format != nullptr, "Null log message");')
        java.append(f'  public static native {jret} {name}({", ".join(jargs)});')
        manifest[name] = {'return': jret, 'arguments': jargs, 'custom': name in CUSTOM}
        if name in CUSTOM:
            continue
        if name == 'HFLogPrint':
            call.insert(1, '"%s"')
        cpp += [f'extern "C" JNIEXPORT {"void" if ret == "void" else "j" + jret} JNICALL {PREFIX}{name}(JNIEnv* env, jclass{", " if cargs else ""}{", ".join(cargs)}) {{',
                '  try { Context context;']
        cpp += ['    ' + line for line in pre]
        cpp.append(f'    {"auto call_status = " if ret != "void" else ""}{name}({", ".join(call)});')
        cpp += ['    ' + line for line in post]
        if ret != 'void':
            cpp.append(f'    return static_cast<j{jret}>(call_status);')
        cpp += ['  } catch (...) { TranslateException(env); }',
                '  return;' if ret == 'void' else '  return HERR_INVALID_PARAM;', '}']
    java += ['}']
    const_java = [f'package {PACKAGE};', '/** Values from the public C API headers. Unsigned 64-bit masks use Java long bits. */',
                  'public final class NativeConstants {', '  private NativeConstants() {}']
    for name, value in constants.items():
        is_long = name.startswith(('HF_CAPTURE_FILTER_', 'HF_CAPTURE_REJECT_'))
        const_java += [f'  public static final {"long" if is_long else "int"} {name} = {value}{"L" if is_long else ""};']
    const_java += ['}']
    java_dir = output / 'java' / PACKAGE.replace('.', '/')
    java_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in [('NativeTypes.java', java_types), ('Native.java', java), ('NativeConstants.java', const_java)]:
        (java_dir / name).write_text('// Generated by portable/generate.py; edit the public C headers or generator.\n' + '\n'.join(lines) + '\n')
    (output / 'bindings.cpp').write_text('\n'.join(cpp) + '\n')
    (output / 'api-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Portable JNI: {len(functions)} C API functions, {len(structs)} descriptors, {len(constants)} constants')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.root.resolve(), arguments.output.resolve())
