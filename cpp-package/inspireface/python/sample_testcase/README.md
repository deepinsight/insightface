# InspireFace Python API sample testcase suite

This is the canonical comprehensive regression suite for the Python wrapper. It follows the C++ Catch2 test organization while reusing the exact same fixtures and model packs under the repository-level `test_res` directory.

## Run

From the repository root:

```bash
PYTHONPATH=python python -m sample_testcase.run \
  --native-lib build_runtime_after/lib/libInspireFace.dylib
```

The command-line paths mirror the C++ runner:

```bash
PYTHONPATH=python python -m sample_testcase.run \
  --test_dir test_res \
  --pack_path test_res/pack/Pikachu \
  --native-lib /path/to/libInspireFace.dylib
```

Use `--benchmark` to enable the extended 192/320/640 detection benchmarks. Use one or more `--pattern` options to select cases by test ID.

The `--native-lib` option makes a temporary package copy and replaces only its native library. It does not overwrite the package library in the working tree. This is useful because a generated Python binding and its native library must have matching ABIs.

## C++ mapping

| Python module | C++ reference |
| --- | --- |
| `test_system_and_stream.py` | `test_system.cpp`, stream lifecycle sections |
| `test_image_process.py` | `test_image_process.cpp`, `test_image_bitmap.cpp` |
| `test_face_track.py` | `test_face_track.cpp`, base track cases |
| `test_pipeline.py` | `test_face_pipeline.cpp` |
| `test_recognition_and_hub.py` | `test_feature_manage.cpp`, `test_feature_hub.cpp`, `test_session_parallel.cpp` |
| `test_similarity_and_errors.py` | `test_similarity_converter.cpp`, Python exception contracts |
| `test_performance.py` | `test_benchmark.cpp` |
| `test_api_coverage.py` | Python public-API and shared-fixture manifest |

## Result policy

- Geometry, buffer, resource, persistence, identity, and API-contract results are hard gates.
- Model classification fields use the same fixtures as C++, but only code-level invariants are gated when the current C++ expectations are already stale for the current model pack.
- Hardware-specific APIs are explicitly listed in the public-API manifest and must be exercised on their matching platform runners.
- JSON results and latency metrics are written under `benchmark_logs/`, which is ignored by Git.

Known wrapper gaps must be represented by failing contract tests while they are being repaired; the canonical suite contains no expected-failure exemptions.
