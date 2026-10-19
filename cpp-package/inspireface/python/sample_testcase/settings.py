"""Shared paths and switches for the Python API sample testcase suite."""

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get(
        "INSPIREFACE_TEST_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()
TEST_RES_DIR = Path(
    os.environ.get("INSPIREFACE_TEST_RES", str(PROJECT_ROOT / "test_res"))
).resolve()
MODEL_PATH = Path(
    os.environ.get(
        "INSPIREFACE_TEST_MODEL",
        str(TEST_RES_DIR / "pack" / "Pikachu"),
    )
).resolve()
REPORT_PATH = Path(
    os.environ.get(
        "INSPIREFACE_TEST_REPORT",
        str(PROJECT_ROOT / "benchmark_logs" / "python_api_test_report.json"),
    )
).resolve()
RUN_BENCHMARKS = os.environ.get("INSPIREFACE_RUN_BENCHMARKS") == "1"


def data_path(relative_path):
    path = TEST_RES_DIR / "data" / relative_path
    if not path.is_file():
        raise FileNotFoundError("Shared C++ test fixture not found: {}".format(path))
    return path
