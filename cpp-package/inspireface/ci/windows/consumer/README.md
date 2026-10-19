# Installed Windows SDK consumer

This standalone project uses only `find_package(InspireFace CONFIG REQUIRED)`
and the installed `InspireFace::InspireFace` target. It does not add SDK include
paths, link dependencies, or DLL import/export definitions itself.

Copy the complete installation to a fresh location, then configure a fresh
consumer build from an MSVC developer shell:

```powershell
cmake -S ci/windows/consumer -B build/windows-consumer -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DInspireFace_DIR="C:/sdk-relocated/InspireFace/lib/cmake/InspireFace" `
  -DISF_CONSUMER_MODEL="C:/test-data/pack/Pikachu"
cmake --build build/windows-consumer --parallel 4
ctest --test-dir build/windows-consumer --output-on-failure
```

Use the same commands with a separately installed static SDK to check its
transitive MNN dependency. The C source is compiled as C and linked with the
C++ driver so that static C++ dependencies can be resolved.

The two model-free tests check C API metadata and exact bitmap contents, C++
image/frame processing, and an InspireCV inline method that accesses exported
static data. They also check that the package's shared/static macros match the
actual imported target and do not expose the SDK producer's `ISF_EXPORTS`.

When `ISF_CONSUMER_MODEL` is supplied, two additional tests load that resource
pack, create sessions, run detection on a small blank image, and release the
resources. Without this argument, model tests are explicitly omitted.

For DLL builds, the SDK DLL is copied beside each executable using the imported
target's installed location. Windows CTest runs receive only System32 in PATH;
they cannot find an SDK DLL through a developer's build-tree PATH. The consumer
build contains no copied source headers or development libraries.
