"""Runner for the comprehensive InspireFace Python API sample testcase suite."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PYTHON_ROOT.parent
TEST_MODULES = (
    "sample_testcase.test_system_and_stream",
    "sample_testcase.test_image_process",
    "sample_testcase.test_face_track",
    "sample_testcase.test_face_capture",
    "sample_testcase.test_pipeline",
    "sample_testcase.test_recognition_and_hub",
    "sample_testcase.test_similarity_and_errors",
    "sample_testcase.test_api_coverage",
    "sample_testcase.test_pythonic_contracts",
    "sample_testcase.test_resource_manager",
    "sample_testcase.test_performance",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_dir",
        "--test-dir",
        dest="test_dir",
        type=Path,
        default=PROJECT_ROOT / "test_res",
        help="Shared test_res directory, matching the C++ test runner",
    )
    parser.add_argument(
        "--pack_path",
        "--pack-path",
        dest="pack_path",
        type=Path,
        default=None,
        help="Model pack path, matching the C++ test runner",
    )
    parser.add_argument(
        "--native-lib",
        type=Path,
        default=None,
        help="Temporarily run against a freshly built native library",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "benchmark_logs" / "python_api_test_report.json",
        help="Ignored JSON result report",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable extended performance cases",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Only run test IDs containing this substring; repeatable",
    )
    parser.add_argument("--verbosity", type=int, default=2)
    return parser.parse_args(argv)


def native_library_destination(package_root):
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        platform_dir = "darwin"
        library_name = "libInspireFace.dylib"
        architecture = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        platform_dir = "linux"
        library_name = "libInspireFace.so"
        architecture = "arm64" if machine in ("arm64", "aarch64") else "x64"
    elif system == "windows":
        platform_dir = "windows"
        library_name = "libInspireFace.dll"
        architecture = "arm64" if machine == "arm64" else "x64"
    else:
        raise RuntimeError("Unsupported native test platform: {}".format(system))
    return package_root / "inspireface" / "modules" / "core" / "libs" / platform_dir / architecture / library_name


def run_with_native_override(args, original_argv):
    native_library = args.native_lib.resolve()
    if not native_library.is_file():
        raise FileNotFoundError("Native library not found: {}".format(native_library))
    with tempfile.TemporaryDirectory(prefix="inspireface-python-suite-") as temp_dir:
        temp_root = Path(temp_dir)
        shutil.copytree(str(PYTHON_ROOT / "inspireface"), str(temp_root / "inspireface"))
        shutil.copy2(str(PYTHON_ROOT / "version.txt"), str(temp_root / "version.txt"))
        destination = native_library_destination(temp_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(native_library), str(destination))

        child_args = []
        skip_next = False
        for argument in original_argv:
            if skip_next:
                skip_next = False
                continue
            if argument == "--native-lib":
                skip_next = True
                continue
            if argument.startswith("--native-lib="):
                continue
            child_args.append(argument)

        environment = os.environ.copy()
        python_path = [str(temp_root), str(PYTHON_ROOT)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        environment["INSPIREFACE_TEST_NATIVE_OVERRIDE"] = str(native_library)
        command = [sys.executable, "-m", "sample_testcase.run"] + child_args
        return subprocess.call(command, cwd=str(PROJECT_ROOT), env=environment)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        unittest.TextTestResult.__init__(self, *args, **kwargs)
        self.records = []
        self._started_at = {}

    def startTest(self, test):
        self._started_at[test] = time.perf_counter()
        unittest.TextTestResult.startTest(self, test)

    def _record(self, test, status, detail=None):
        started = self._started_at.pop(test, time.perf_counter())
        record = {
            "id": test.id(),
            "status": status,
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }
        if detail:
            record["detail"] = detail
        self.records.append(record)

    def addSuccess(self, test):
        self._record(test, "passed")
        unittest.TextTestResult.addSuccess(self, test)

    def addFailure(self, test, error):
        self._record(test, "failed", self._exc_info_to_string(error, test))
        unittest.TextTestResult.addFailure(self, test, error)

    def addError(self, test, error):
        self._record(test, "error", self._exc_info_to_string(error, test))
        unittest.TextTestResult.addError(self, test, error)

    def addSkip(self, test, reason):
        self._record(test, "skipped", reason)
        unittest.TextTestResult.addSkip(self, test, reason)

    def addExpectedFailure(self, test, error):
        self._record(test, "expected_failure", self._exc_info_to_string(error, test))
        unittest.TextTestResult.addExpectedFailure(self, test, error)

    def addUnexpectedSuccess(self, test):
        self._record(test, "unexpected_success")
        unittest.TextTestResult.addUnexpectedSuccess(self, test)


class RecordingRunner(unittest.TextTestRunner):
    resultclass = RecordingResult


def filter_suite(suite, patterns):
    if not patterns:
        return suite
    filtered = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            nested = filter_suite(item, patterns)
            if nested.countTestCases():
                filtered.addTest(nested)
        elif any(pattern in item.id() for pattern in patterns):
            filtered.addTest(item)
    return filtered


def write_report(path, result, native_version, elapsed_ms):
    from sample_testcase.common import collected_metrics

    payload = {
        "native_version": native_version,
        "native_override": os.environ.get("INSPIREFACE_TEST_NATIVE_OVERRIDE"),
        "summary": {
            "run": result.testsRun,
            "passed": sum(record["status"] == "passed" for record in result.records),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses),
            "elapsed_ms": elapsed_ms,
            "successful": result.wasSuccessful(),
        },
        "tests": result.records,
        "metrics": collected_metrics(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    if args.native_lib is not None and not os.environ.get("INSPIREFACE_TEST_NATIVE_OVERRIDE"):
        return run_with_native_override(args, raw_argv)

    test_dir = args.test_dir.resolve()
    pack_path = (
        args.pack_path.resolve()
        if args.pack_path is not None
        else test_dir / "pack" / "Pikachu"
    )
    if not test_dir.is_dir():
        raise FileNotFoundError("test_res directory not found: {}".format(test_dir))
    if not pack_path.is_file():
        raise FileNotFoundError("Model pack not found: {}".format(pack_path))

    os.environ["INSPIREFACE_TEST_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["INSPIREFACE_TEST_RES"] = str(test_dir)
    os.environ["INSPIREFACE_TEST_MODEL"] = str(pack_path)
    os.environ["INSPIREFACE_TEST_REPORT"] = str(args.report.resolve())
    os.environ["INSPIREFACE_RUN_BENCHMARKS"] = "1" if args.benchmark else "0"

    import inspireface as ifac

    native_version = ifac.version()
    version_tuple = tuple(int(part) for part in native_version.split("."))
    if version_tuple < (1, 2, 3):
        raise RuntimeError(
            "Native library {} is older than the Python binding ABI. "
            "Use --native-lib with a current build.".format(native_version)
        )

    ifac.launch(resource_path=str(pack_path))
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for module_name in TEST_MODULES:
        module = __import__(module_name, fromlist=["*"])
        suite.addTests(loader.loadTestsFromModule(module))
    suite = filter_suite(suite, args.pattern)

    started = time.perf_counter()
    try:
        runner = RecordingRunner(verbosity=args.verbosity)
        result = runner.run(suite)
    finally:
        if ifac.query_launch_status():
            ifac.terminate()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    write_report(args.report.resolve(), result, native_version, elapsed_ms)
    print("JSON report: {}".format(args.report.resolve()))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
