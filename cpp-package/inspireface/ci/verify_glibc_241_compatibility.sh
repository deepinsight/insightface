#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <libInspireFace.so>" >&2
    exit 2
fi

library_path=$1
if [[ ! -f "${library_path}" ]]; then
    echo "Shared library does not exist: ${library_path}" >&2
    exit 2
fi

glibc_version=$(ldd --version | sed -n '1p')
if [[ ! "${glibc_version}" =~ 2\.41([^0-9.]|$) ]]; then
    echo "Expected glibc 2.41, found: ${glibc_version}" >&2
    exit 1
fi
echo "glibc_version=${glibc_version}"

stack_header=$(readelf -W -l "${library_path}" | grep 'GNU_STACK' || true)
if [[ -z "${stack_header}" ]]; then
    echo "Missing GNU_STACK program header: ${library_path}" >&2
    exit 1
fi
if [[ "${stack_header}" == *E* ]]; then
    echo "Executable stack detected: ${stack_header}" >&2
    exit 1
fi
echo "stack_header=${stack_header}"

if [[ -n "${GLIBC_TUNABLES:-}" ]]; then
    echo "GLIBC_TUNABLES must not mask the default loader behavior" >&2
    exit 1
fi

for iteration in $(seq 1 10); do
    result=$(ISF_TEST_LIBRARY="${library_path}" python3 -c '
import ctypes
import os
import time

path = os.environ["ISF_TEST_LIBRARY"]
start = time.perf_counter()
ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
elapsed_ms = (time.perf_counter() - start) * 1000
print(f"load_result=success elapsed_ms={elapsed_ms:.3f}")
')
    echo "iteration=${iteration} ${result}"
done

echo "glibc_2_41_compatibility=passed load_attempts=10"
