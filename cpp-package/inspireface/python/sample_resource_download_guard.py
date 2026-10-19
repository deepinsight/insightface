#!/usr/bin/env python3
"""Network-free correctness and latency gate for secure model downloads."""

import hashlib
import importlib.util
import ssl
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock


RESOURCE_MODULE_PATH = (
    Path(__file__).resolve().parent
    / "inspireface"
    / "modules"
    / "utils"
    / "resource.py"
)
MODEL_NAME = "GuardModel"


def load_resource_module():
    spec = importlib.util.spec_from_file_location(
        "inspireface_resource_guard_target", RESOURCE_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {RESOURCE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload, fail_after=None, observer=None):
        self.payload = payload
        self.fail_after = fail_after
        self.observer = observer
        self.offset = 0
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, block_size):
        if self.observer is not None:
            self.observer()
        if self.fail_after is not None and self.offset >= self.fail_after:
            raise ConnectionResetError("simulated interrupted download")
        if self.offset >= len(self.payload):
            return b""

        end = min(self.offset + block_size, len(self.payload))
        if self.fail_after is not None:
            end = min(end, self.fail_after)
        block = self.payload[self.offset:end]
        self.offset = end
        return block


class FakeUrlOpen:
    def __init__(self, payload=b"", error=None, fail_after=None, observer=None):
        self.payload = payload
        self.error = error
        self.fail_after = fail_after
        self.observer = observer
        self.calls = []

    def __call__(self, request, context=None, timeout=None):
        self.calls.append(
            {"request": request, "context": context, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return FakeResponse(
            self.payload,
            fail_after=self.fail_after,
            observer=self.observer,
        )


class Guard:
    def __init__(self):
        self.groups = {
            "tls_verification": True,
            "byte_exact": True,
            "atomic_cleanup": True,
            "fault_propagation": True,
            "cache_behavior": True,
            "latency_gate": True,
        }
        self.failures = []

    def expect(self, condition, group, message):
        if condition:
            return
        self.groups[group] = False
        self.failures.append(f"{group}: {message}")

    def passed(self):
        return not self.failures


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def make_manager(resource_module, home_dir, expected_payload):
    home_dir.mkdir()
    with mock.patch.object(resource_module.Path, "home", return_value=home_dir):
        manager = resource_module.ResourceManager(use_modelscope=False)
    manager._MODEL_LIST = {
        MODEL_NAME: {
            "url": "https://models.example.invalid/GuardModel",
            "filename": "GuardModel",
            "sha256": sha256(expected_payload),
        }
    }
    return manager


def model_path(manager):
    return manager.models_dir / manager._MODEL_LIST[MODEL_NAME]["filename"]


def has_download_artifacts(manager):
    destination = model_path(manager)
    downloading_flag = destination.with_suffix(".downloading")
    partial_files = list(manager.models_dir.glob(f".{destination.name}.*.part"))
    return downloading_flag.exists() or bool(partial_files)


def expect_download_error(guard, resource_module, manager, fake_open, expected_cause_type):
    caught = None
    with mock.patch.object(resource_module.urllib.request, "urlopen", fake_open):
        try:
            manager.get_model(MODEL_NAME)
        except RuntimeError as error:
            caught = error
    guard.expect(caught is not None, "fault_propagation", "download failure was not propagated")
    if caught is not None:
        guard.expect(
            isinstance(caught.__cause__, expected_cause_type),
            "fault_propagation",
            f"expected cause {expected_cause_type.__name__}, got {type(caught.__cause__).__name__}",
        )
    return caught


def main():
    resource_module = load_resource_module()
    guard = Guard()
    started = time.perf_counter()

    payloads = [
        b"small-model-payload",
        bytes(range(256)) * 37,
        b"\x00\xff\x55\xaa" * 4096,
        hashlib.sha256(b"deterministic-model").digest() * 8192,
    ]

    with tempfile.TemporaryDirectory(prefix="inspireface-download-guard-") as temp_dir:
        root = Path(temp_dir)

        # Multiple byte patterns must survive download and atomic replacement exactly.
        last_manager = None
        for index, payload in enumerate(payloads):
            manager = make_manager(resource_module, root / f"success-{index}", payload)
            fake_open = FakeUrlOpen(payload)
            with mock.patch.object(resource_module.urllib.request, "urlopen", fake_open):
                result = Path(manager.get_model(MODEL_NAME))
            guard.expect(result.read_bytes() == payload, "byte_exact", f"payload {index} changed")
            guard.expect(not has_download_artifacts(manager), "atomic_cleanup", f"payload {index} left artifacts")
            guard.expect(len(fake_open.calls) == 1, "cache_behavior", f"payload {index} download count")
            if fake_open.calls:
                call = fake_open.calls[0]
                context = call["context"]
                guard.expect(
                    context is not None
                    and context.check_hostname
                    and context.verify_mode == ssl.CERT_REQUIRED,
                    "tls_verification",
                    f"payload {index} used an unverified TLS context",
                )
                guard.expect(
                    call["timeout"] == manager.DOWNLOAD_TIMEOUT_SECONDS,
                    "tls_verification",
                    f"payload {index} did not set the download timeout",
                )
            last_manager = manager

        # A valid cached model must not access the network.
        forbidden_open = FakeUrlOpen(error=AssertionError("cache accessed network"))
        with mock.patch.object(resource_module.urllib.request, "urlopen", forbidden_open):
            cached = Path(last_manager.get_model(MODEL_NAME))
        guard.expect(cached.read_bytes() == payloads[-1], "cache_behavior", "cached bytes changed")
        guard.expect(not forbidden_open.calls, "cache_behavior", "valid cache accessed network")

        # A hash mismatch must fail closed and remove all temporary state.
        expected_payload = b"expected-model"
        manager = make_manager(resource_module, root / "hash-mismatch", expected_payload)
        mismatch_error = expect_download_error(
            guard,
            resource_module,
            manager,
            FakeUrlOpen(b"tampered-model"),
            RuntimeError,
        )
        guard.expect(not model_path(manager).exists(), "byte_exact", "hash mismatch published a model")
        guard.expect(not has_download_artifacts(manager), "atomic_cleanup", "hash mismatch left artifacts")
        if mismatch_error is not None:
            guard.expect(
                "SHA-256 mismatch" in str(mismatch_error.__cause__),
                "fault_propagation",
                "hash mismatch details were lost",
            )

        # TLS verification failures must retain their cause and leave no files behind.
        manager = make_manager(resource_module, root / "tls-error", expected_payload)
        expect_download_error(
            guard,
            resource_module,
            manager,
            FakeUrlOpen(error=ssl.SSLCertVerificationError("simulated certificate failure")),
            ssl.SSLCertVerificationError,
        )
        guard.expect(not model_path(manager).exists(), "tls_verification", "TLS failure published a model")
        guard.expect(not has_download_artifacts(manager), "atomic_cleanup", "TLS failure left artifacts")

        # An interrupted refresh must never expose partial bytes or delete the old model.
        replacement = b"replacement-model" * 2048
        manager = make_manager(resource_module, root / "interrupted", replacement)
        destination = model_path(manager)
        previous = b"previous-valid-model"
        destination.write_bytes(previous)
        observed_old_file = []

        def observe_destination():
            observed_old_file.append(destination.exists() and destination.read_bytes() == previous)

        interrupted_open = FakeUrlOpen(
            replacement,
            fail_after=4096,
            observer=observe_destination,
        )
        caught = None
        with mock.patch.object(resource_module.urllib.request, "urlopen", interrupted_open):
            try:
                manager.get_model(MODEL_NAME, re_download=True)
            except RuntimeError as error:
                caught = error
        guard.expect(caught is not None, "fault_propagation", "interrupted download succeeded")
        guard.expect(destination.read_bytes() == previous, "atomic_cleanup", "interrupted download replaced old model")
        guard.expect(all(observed_old_file), "atomic_cleanup", "partial data became externally visible")
        guard.expect(not has_download_artifacts(manager), "atomic_cleanup", "interrupted download left artifacts")

        # A successful refresh must expose the old or new complete file, never a partial file.
        manager = make_manager(resource_module, root / "atomic-refresh", replacement)
        destination = model_path(manager)
        destination.write_bytes(previous)
        observed_complete_file = []

        def observe_complete_destination():
            observed_complete_file.append(destination.read_bytes() == previous)

        refresh_open = FakeUrlOpen(replacement, observer=observe_complete_destination)
        with mock.patch.object(resource_module.urllib.request, "urlopen", refresh_open):
            refreshed = Path(manager.get_model(MODEL_NAME, re_download=True))
        guard.expect(refreshed.read_bytes() == replacement, "byte_exact", "refresh bytes changed")
        guard.expect(
            all(observed_complete_file),
            "atomic_cleanup",
            "successful refresh exposed a partial model",
        )
        guard.expect(not has_download_artifacts(manager), "atomic_cleanup", "successful refresh left artifacts")

        # A corrupted cache must be replaced, while explicit hash bypass remains available.
        manager = make_manager(resource_module, root / "corrupt-cache", expected_payload)
        model_path(manager).write_bytes(b"corrupt-cache")
        fake_open = FakeUrlOpen(expected_payload)
        with mock.patch.object(resource_module.urllib.request, "urlopen", fake_open):
            result = Path(manager.get_model(MODEL_NAME))
        guard.expect(result.read_bytes() == expected_payload, "cache_behavior", "corrupt cache was not repaired")

        manager = make_manager(resource_module, root / "ignore-hash", expected_payload)
        unverified_payload = b"explicitly-unverified-model"
        fake_open = FakeUrlOpen(unverified_payload)
        with mock.patch.object(resource_module.urllib.request, "urlopen", fake_open):
            result = Path(manager.get_model(MODEL_NAME, ignore_verification=True))
        guard.expect(result.read_bytes() == unverified_payload, "byte_exact", "hash bypass changed bytes")
        if fake_open.calls:
            context = fake_open.calls[0]["context"]
            guard.expect(
                context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED,
                "tls_verification",
                "hash bypass also disabled TLS verification",
            )

        # Record cached SHA-256 cost separately from download/network latency.
        performance_payload = bytes(range(256)) * (4 * 1024 * 1024 // 256)
        performance_file = root / "hash-performance.bin"
        performance_file.write_bytes(performance_payload)
        resource_module.get_file_hash_sha256(performance_file)
        samples_ms = []
        for _ in range(7):
            hash_started = time.perf_counter()
            digest = resource_module.get_file_hash_sha256(performance_file)
            samples_ms.append((time.perf_counter() - hash_started) * 1000.0)
            guard.expect(digest == sha256(performance_payload), "byte_exact", "performance hash mismatch")
        hash_p50_ms = statistics.median(samples_ms)
        throughput_mib_s = 4.0 / (hash_p50_ms / 1000.0)
        guard.expect(throughput_mib_s >= 20.0, "latency_gate", "SHA-256 throughput below 20 MiB/s")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(f"payloads={len(payloads)}")
    for group, passed in guard.groups.items():
        print(f"{group}={'PASS' if passed else 'FAIL'}")
    print(f"sha256_p50_ms_4mib={hash_p50_ms:.3f}")
    print(f"sha256_throughput_mib_s={throughput_mib_s:.1f}")
    print(f"total_elapsed_ms={elapsed_ms:.3f}")
    for failure in guard.failures:
        print(f"failure={failure}")
    return 0 if guard.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
