"""Persistent, signature-scoped CoreML session compilation caches.

CoreML compiles ONNX Runtime partitions while an ``InferenceSession`` is
constructed.  ``ModelCacheDirectory`` makes those compiled artifacts survive
the process, but ONNX Runtime's own cache key does not describe every Session
contract used by InsightFace.  This module therefore gives every distinct
model/input/provider contract its own cache directory and remembers which
CoreML compute-unit policy successfully created the Session.

The module deliberately has no dependency on InsightFace model classes.  It
accepts an InferenceSession-compatible factory, which also keeps its behavior
straightforward to test without invoking CoreML.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

try:  # ``fcntl`` is available on macOS, but importing this module is portable.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - exercised on Windows.
    fcntl = None


COREML_PROVIDER = "CoreMLExecutionProvider"
CACHE_SCHEMA = 1
DEFAULT_COMPUTE_UNITS = "ALL"
FALLBACK_COMPUTE_UNITS = "CPUAndGPU"

_MARKER_NAME = ".insightface-validated.json"
_SELECTION_NAME = "selection.json"
_SIGNATURE_NAME = "signature.json"
_LOCK_NAME = ".selection.lock"
_RESERVED_SESSION_KWARGS = frozenset(
    {"providers", "provider_options", "sess_options"}
)
_COPYABLE_SESSION_OPTION_ATTRIBUTES = (
    "enable_cpu_mem_arena",
    "enable_mem_pattern",
    "enable_mem_reuse",
    "enable_profiling",
    "execution_mode",
    "execution_order",
    "graph_optimization_level",
    "inter_op_num_threads",
    "intra_op_num_threads",
    "log_severity_level",
    "log_verbosity_level",
    "logid",
    "profile_file_prefix",
    "use_deterministic_compute",
    "use_per_session_threads",
)

_thread_locks_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}


class CoreMLSessionError(RuntimeError):
    """Raised after every permitted CoreML Session candidate has failed."""

    def __init__(self, attempts: Sequence[tuple[str, Exception]]):
        self.attempts = tuple(attempts)
        summary = "; ".join(
            f"{compute_units}: {type(error).__name__}: {error}"
            for compute_units, error in self.attempts
        )
        super().__init__(f"failed to create a CoreML session ({summary})")


@dataclass(frozen=True)
class CoreMLSessionResult:
    """A created Session plus the cache decision used to create it."""

    session: Any
    compute_units: Optional[str]
    cache_directory: Optional[Path]
    cache_hit: bool
    warmup_performed: bool
    base_signature: Optional[Mapping[str, Any]]
    signature: Optional[Mapping[str, Any]]
    selection_metadata: Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class _Candidate:
    compute_units: str
    signature: Mapping[str, Any]
    signature_hash: str
    cache_directory: Path


def default_coreml_cache_root() -> Path:
    """Return InsightFace's platform-safe, user-writable CoreML cache root."""

    return Path.home() / ".insightface" / "cache" / "coreml" / "v1"


def _new_session_options() -> Any:
    # Kept behind a tiny function so tests can supply a fake SessionOptions
    # implementation without importing or initializing ONNX Runtime.
    import onnxruntime

    return onnxruntime.SessionOptions()


def copy_session_options(
    source: Any = None,
    dimension_overrides: Optional[Mapping[str, int]] = None,
) -> Any:
    """Create safe, fresh SessionOptions with optional named dimensions.

    ``source`` may be either a SessionOptions instance or an InferenceSession
    exposing ``get_session_options``.  ONNX Runtime does not expose custom-op
    registrations, external initializers, or arbitrary config entries for
    copying, so only its documented mutable scalar attributes are preserved.
    A new object is always returned, which is important when separate CoreML
    candidates or fixed-resolution Sessions are constructed.
    """

    target = _new_session_options()
    getter = getattr(source, "get_session_options", None)
    if callable(getter):
        source = getter()
    if source is not None:
        for name in _COPYABLE_SESSION_OPTION_ATTRIBUTES:
            try:
                setattr(target, name, getattr(source, name))
            except (AttributeError, TypeError, ValueError):
                continue

    for name, value in dict(dimension_overrides or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("dimension override names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("dimension override values must be integers")
        value = int(value)
        if value <= 0:
            raise ValueError("dimension override values must be positive")
        target.add_free_dimension_override_by_name(name, value)
    return target


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _signature_hash(signature: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(signature)).hexdigest()


def _normalize_sha256(value: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError("model_sha256 must be a 64-character hexadecimal SHA256")
    return result


def _normalize_input_contracts(
    input_contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(input_contracts, (str, bytes, Mapping)):
        raise TypeError("input_contracts must be a sequence of mappings")
    result = []
    for index, contract in enumerate(input_contracts):
        if not isinstance(contract, Mapping):
            raise TypeError(f"input_contracts[{index}] must be a mapping")
        if set(contract) != {"name", "dtype", "shape"}:
            raise ValueError(
                f"input_contracts[{index}] keys must be exactly "
                "['dtype', 'name', 'shape']"
            )
        name = contract["name"]
        dtype = contract["dtype"]
        shape = contract["shape"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"input_contracts[{index}].name must be non-empty")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"input_contracts[{index}].dtype must be non-empty")
        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"input_contracts[{index}].shape must be a sequence")
        normalized_shape = []
        for dimension in shape:
            if isinstance(dimension, np.integer):
                dimension = int(dimension)
            if isinstance(dimension, bool) or not (
                dimension is None
                or isinstance(dimension, (int, str))
            ):
                raise TypeError(
                    f"input_contracts[{index}].shape dimensions must be "
                    "integers, strings, or null"
                )
            if isinstance(dimension, str) and not dimension:
                raise ValueError("symbolic input dimensions must be non-empty")
            normalized_shape.append(dimension)
        result.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": normalized_shape,
            }
        )
    if not result:
        raise ValueError("input_contracts must contain at least one input")
    return result


def _normalize_provider_options(
    providers: Sequence[str],
    provider_options: Any,
) -> list[dict[str, Any]]:
    if provider_options is None:
        return [{} for _provider in providers]
    if isinstance(provider_options, Mapping):
        # Accept either the mapping returned by get_provider_options(), or a
        # single provider's direct options for the one-provider case.
        if all(
            provider in provider_options
            and isinstance(provider_options[provider], Mapping)
            for provider in providers
        ):
            return [dict(provider_options[provider]) for provider in providers]
        if len(providers) == 1:
            return [dict(provider_options)]
        raise TypeError(
            "provider_options mappings must be keyed by provider name"
        )
    if isinstance(provider_options, (str, bytes)):
        raise TypeError("provider_options must be a mapping or sequence")
    values = [dict(value or {}) for value in provider_options]
    if len(values) > len(providers):
        raise ValueError("provider_options has more entries than providers")
    values.extend({} for _ in range(len(providers) - len(values)))
    return values


def _default_diagnostics() -> dict[str, str]:
    try:
        ort_version = importlib_metadata.version("onnxruntime")
    except importlib_metadata.PackageNotFoundError:
        try:
            ort_version = importlib_metadata.version("onnxruntime-gpu")
        except importlib_metadata.PackageNotFoundError:
            ort_version = "unknown"
    values = {
        "onnxruntime": ort_version,
        "system": platform.system(),
        "system_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    macos = platform.mac_ver()[0]
    if macos:
        values["macos"] = macos
    return values


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(str(path)))
    with _thread_locks_guard:
        return _thread_locks.setdefault(key, threading.RLock())


@contextmanager
def _selection_lock(base_directory: Path):
    """Serialize cache probing and compilation within a signature."""

    base_directory.mkdir(parents=True, exist_ok=True)
    lock = _thread_lock(base_directory)
    with lock:
        lock_path = base_directory / _LOCK_NAME
        with lock_path.open("a+b") as stream:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _clear_candidate(candidate: _Candidate) -> None:
    # The directory is entirely hash-derived beneath the managed base, so
    # clearing it cannot escape to a caller-controlled model path.
    shutil.rmtree(str(candidate.cache_directory), ignore_errors=True)


def _marker_valid(candidate: _Candidate) -> bool:
    marker = _read_json(candidate.cache_directory / _MARKER_NAME)
    return bool(
        marker
        and marker.get("schema") == CACHE_SCHEMA
        and marker.get("signature_sha256") == candidate.signature_hash
        and marker.get("compute_units") == candidate.compute_units
    )


def _has_cache_artifact(candidate: _Candidate) -> bool:
    """Return whether ORT left data beyond our own small metadata files."""

    try:
        return any(
            path.name not in {_MARKER_NAME, _SIGNATURE_NAME}
            for path in candidate.cache_directory.iterdir()
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        return False


def _write_signature(path: Path, signature: Mapping[str, Any]) -> None:
    _atomic_write_json(
        path,
        {
            "schema": CACHE_SCHEMA,
            "signature_sha256": _signature_hash(signature),
            "signature": dict(signature),
        },
    )


def _selection_candidate(
    selection: Optional[Mapping[str, Any]],
    base_hash: str,
    candidates: Sequence[_Candidate],
    selection_policy: str,
) -> Optional[_Candidate]:
    if not selection or selection.get("schema") != CACHE_SCHEMA:
        return None
    if selection.get("base_signature_sha256") != base_hash:
        return None
    if selection.get("selection_policy") != selection_policy:
        return None
    for candidate in candidates:
        if (
            selection.get("compute_units") == candidate.compute_units
            and selection.get("signature_sha256") == candidate.signature_hash
        ):
            return candidate
    return None


_NUMPY_DTYPES = {
    "bool": np.bool_,
    "double": np.float64,
    "float": np.float32,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "tensor(bool)": np.bool_,
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


def _zero_warmup_feed(
    input_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    feed = {}
    for contract in input_contracts:
        dtype_name = str(contract["dtype"]).strip().lower()
        try:
            dtype = _NUMPY_DTYPES[dtype_name]
        except KeyError as error:
            raise ValueError(
                f"zero warmup does not support input dtype {contract['dtype']!r}"
            ) from error
        shape = tuple(
            int(dimension)
            if isinstance(dimension, int) and dimension > 0
            else 1
            for dimension in contract["shape"]
        )
        feed[str(contract["name"])] = np.zeros(shape, dtype=dtype)
    return feed


def _prepare_warmup(
    warmup: Any,
    input_contracts: Sequence[Mapping[str, Any]],
) -> Optional[Callable[[Any], None]]:
    if warmup is None or warmup is False:
        return None
    if warmup is True:
        feed = _zero_warmup_feed(input_contracts)

        def run_zero_warmup(session: Any) -> None:
            runner = getattr(session, "run", None)
            if not callable(runner):
                raise TypeError("zero warmup requires a session.run method")
            runner(None, feed)

        return run_zero_warmup
    if isinstance(warmup, Mapping):
        feed = dict(warmup)

        def run_feed_warmup(session: Any) -> None:
            runner = getattr(session, "run", None)
            if not callable(runner):
                raise TypeError("feed warmup requires a session.run method")
            runner(None, feed)

        return run_feed_warmup
    if callable(warmup):
        return warmup
    raise TypeError("warmup must be false, true, a feed mapping, or a callback")


def _validate_coreml_primary(session: Any) -> None:
    getter = getattr(session, "get_providers", None)
    if not callable(getter):
        return
    providers = list(getter() or ())
    if not providers or providers[0] != COREML_PROVIDER:
        raise RuntimeError(
            "CoreML candidate silently fell back; the created Session's "
            f"primary provider is {providers[0] if providers else None!r}"
        )


def _invoke_session_factory(
    session_factory: Callable[..., Any],
    model_source: Any,
    providers: Sequence[str],
    provider_options: Sequence[Mapping[str, Any]],
    sess_options_factory: Optional[Callable[[], Any]],
    session_kwargs: Mapping[str, Any],
) -> Any:
    kwargs = dict(session_kwargs)
    kwargs["providers"] = list(providers)
    kwargs["provider_options"] = [dict(value) for value in provider_options]
    if sess_options_factory is not None:
        kwargs["sess_options"] = sess_options_factory()
    return session_factory(model_source, **kwargs)


def _selection_metadata(
    base_hash: str,
    candidate: _Candidate,
    diagnostics: Mapping[str, Any],
    warmup_performed: bool,
    selection_policy: str,
) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "base_signature_sha256": base_hash,
        "signature_sha256": candidate.signature_hash,
        "compute_units": candidate.compute_units,
        "selection_policy": selection_policy,
        "cache_directory": candidate.cache_directory.name,
        "warmup_performed": bool(warmup_performed),
        # Diagnostic versions deliberately do not participate in either hash.
        "created_with": dict(diagnostics),
    }


def create_coreml_session(
    session_factory: Callable[..., Any],
    model_source: Any,
    *,
    providers: Sequence[str],
    provider_options: Any = None,
    model_sha256: str,
    task: str,
    graph_variant: str,
    input_contracts: Sequence[Mapping[str, Any]],
    sess_options_factory: Optional[Callable[[], Any]] = None,
    warmup: Any = False,
    cache_root: Optional[os.PathLike[str] | str] = None,
    session_kwargs: Optional[Mapping[str, Any]] = None,
    diagnostic_metadata: Optional[Mapping[str, Any]] = None,
) -> CoreMLSessionResult:
    """Create an InferenceSession with persistent, safe CoreML fallback.

    ``input_contracts`` is an ordered sequence of mappings whose keys are
    exactly ``name``, ``dtype``, and ``shape``.  Symbolic, null, zero, and
    negative dimensions are represented as size one by automatic zero-input
    warmup.  ``warmup`` may be ``True`` for that automatic feed, a feed
    mapping, a ``callback(session)``, or false to validate construction only.

    When CoreML is absent the factory is called once with the supplied
    providers and no cache policy.  With CoreML, an unspecified or ``ALL``
    compute-unit option attempts ``ALL`` and then ``CPUAndGPU``.  Any explicit
    non-ALL option is respected as the sole candidate.
    """

    if not callable(session_factory):
        raise TypeError("session_factory must be callable")
    providers = [str(provider) for provider in providers]
    options = _normalize_provider_options(providers, provider_options)
    extra_kwargs = dict(session_kwargs or {})
    reserved = _RESERVED_SESSION_KWARGS.intersection(extra_kwargs)
    if reserved:
        raise ValueError(
            f"session_kwargs must not contain reserved keys {sorted(reserved)}"
        )

    # CoreML cache and compute-unit selection only apply when CoreML is the
    # requested primary provider.  A later CoreML entry is merely a fallback
    # behind a different execution provider and must remain untouched.
    if not providers or providers[0] != COREML_PROVIDER:
        session = _invoke_session_factory(
            session_factory,
            model_source,
            providers,
            options,
            sess_options_factory,
            extra_kwargs,
        )
        return CoreMLSessionResult(
            session=session,
            compute_units=None,
            cache_directory=None,
            cache_hit=False,
            warmup_performed=False,
            base_signature=None,
            signature=None,
            selection_metadata=None,
        )

    model_sha256 = _normalize_sha256(model_sha256)
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if not isinstance(graph_variant, str) or not graph_variant.strip():
        raise ValueError("graph_variant must be a non-empty string")
    contracts = _normalize_input_contracts(input_contracts)
    warmup_callback = _prepare_warmup(warmup, contracts)
    diagnostics = dict(_default_diagnostics())
    diagnostics.update(dict(diagnostic_metadata or {}))

    coreml_index = providers.index(COREML_PROVIDER)
    coreml_options = dict(options[coreml_index])
    configured_root = coreml_options.pop("ModelCacheDirectory", None)
    requested_compute_units = coreml_options.pop("MLComputeUnits", None)
    if requested_compute_units in (None, "", DEFAULT_COMPUTE_UNITS):
        compute_candidates = [DEFAULT_COMPUTE_UNITS, FALLBACK_COMPUTE_UNITS]
        selection_policy = "automatic"
    else:
        compute_candidates = [str(requested_compute_units)]
        selection_policy = "explicit"

    base_signature = {
        "schema": CACHE_SCHEMA,
        "model_sha256": model_sha256,
        "task": task.strip(),
        "graph_variant": graph_variant.strip(),
        "inputs": contracts,
        "provider": COREML_PROVIDER,
        "coreml_options": coreml_options,
    }
    base_hash = _signature_hash(base_signature)
    root = Path(
        cache_root
        if cache_root is not None
        else configured_root
        if configured_root
        else default_coreml_cache_root()
    ).expanduser()
    base_directory = root / model_sha256 / base_hash
    candidates = []
    for compute_units in compute_candidates:
        signature = dict(base_signature)
        signature["coreml_options"] = {
            **coreml_options,
            "MLComputeUnits": compute_units,
        }
        signature_hash = _signature_hash(signature)
        candidates.append(
            _Candidate(
                compute_units=compute_units,
                signature=signature,
                signature_hash=signature_hash,
                cache_directory=base_directory / signature_hash,
            )
        )

    selection_path = base_directory / _SELECTION_NAME
    failures: list[tuple[str, Exception]] = []

    def invoke(candidate: _Candidate) -> Any:
        _write_signature(
            candidate.cache_directory / _SIGNATURE_NAME,
            candidate.signature,
        )
        candidate_options = [dict(value) for value in options]
        candidate_options[coreml_index] = {
            **coreml_options,
            "MLComputeUnits": candidate.compute_units,
            "ModelCacheDirectory": str(candidate.cache_directory),
        }
        session = _invoke_session_factory(
            session_factory,
            model_source,
            providers,
            candidate_options,
            sess_options_factory,
            extra_kwargs,
        )
        _validate_coreml_primary(session)
        return session

    def finish(
        candidate: _Candidate,
        session: Any,
        *,
        cache_hit: bool,
        warmup_performed: bool,
    ) -> CoreMLSessionResult:
        metadata = _selection_metadata(
            base_hash,
            candidate,
            diagnostics,
            warmup_performed,
            selection_policy,
        )
        if not cache_hit:
            marker = {
                "schema": CACHE_SCHEMA,
                "signature_sha256": candidate.signature_hash,
                "compute_units": candidate.compute_units,
                "created_with": dict(diagnostics),
                "warmup_performed": bool(warmup_performed),
            }
            _atomic_write_json(
                candidate.cache_directory / _MARKER_NAME,
                marker,
            )
            _atomic_write_json(selection_path, metadata)
        return CoreMLSessionResult(
            session=session,
            compute_units=candidate.compute_units,
            cache_directory=candidate.cache_directory,
            cache_hit=cache_hit,
            warmup_performed=warmup_performed,
            base_signature=base_signature,
            signature=candidate.signature,
            selection_metadata=metadata,
        )

    def build_candidate(
        candidate: _Candidate,
        *,
        existing_cache: bool,
    ) -> CoreMLSessionResult:
        # A failed load from an existing leaf gets one clean rebuild using the
        # same compute policy.  A newly compiled candidate goes directly to
        # the next permitted compute policy on failure.
        attempts = 2 if existing_cache else 1
        for attempt in range(attempts):
            if attempt:
                _clear_candidate(candidate)
                candidate.cache_directory.mkdir(parents=True, exist_ok=True)
            try:
                session = invoke(candidate)
                performed = warmup_callback is not None
                if warmup_callback is not None:
                    warmup_callback(session)
                return finish(
                    candidate,
                    session,
                    cache_hit=False,
                    warmup_performed=performed,
                )
            except Exception as error:
                failures.append((candidate.compute_units, error))
        _clear_candidate(candidate)
        raise failures[-1][1]

    with _selection_lock(base_directory):
        _write_signature(
            base_directory / _SIGNATURE_NAME,
            base_signature,
        )
        selection = _read_json(selection_path)
        selected_candidate = _selection_candidate(
            selection,
            base_hash,
            candidates,
            selection_policy,
        )
        if selection is not None and selected_candidate is None:
            _remove_file(selection_path)

        if selected_candidate is not None:
            selected_existing = _has_cache_artifact(selected_candidate)
            selected_valid = (
                selected_existing and _marker_valid(selected_candidate)
            )
            if selected_valid:
                try:
                    session = invoke(selected_candidate)
                    return finish(
                        selected_candidate,
                        session,
                        cache_hit=True,
                        warmup_performed=False,
                    )
                except Exception as error:
                    warnings.warn(
                        "validated CoreML cache could not be loaded; "
                        "clearing and recompiling the affected signature "
                        f"({type(error).__name__}: {error})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    failures.append((selected_candidate.compute_units, error))
                    _clear_candidate(selected_candidate)
                    _remove_file(selection_path)
                    selected_existing = False
            try:
                return build_candidate(
                    selected_candidate,
                    existing_cache=selected_existing,
                )
            except Exception:
                _remove_file(selection_path)
                # A remembered CPUAndGPU choice avoids retrying a known-bad
                # ALL plan while it remains usable. If that selected plan can
                # no longer be loaded or rebuilt (for example after a runtime
                # upgrade), the normal candidate loop may probe ALL again.

        attempted = {
            selected_candidate.compute_units
            if selected_candidate is not None
            else None
        }
        for candidate in candidates:
            if candidate.compute_units in attempted:
                continue
            existing = _has_cache_artifact(candidate)
            candidate.cache_directory.mkdir(parents=True, exist_ok=True)
            try:
                return build_candidate(
                    candidate,
                    existing_cache=existing,
                )
            except Exception:
                continue

    if failures:
        raise CoreMLSessionError(failures) from failures[-1][1]
    raise CoreMLSessionError(
        [("unknown", RuntimeError("no CoreML compute candidate was available"))]
    )


__all__ = [
    "CACHE_SCHEMA",
    "COREML_PROVIDER",
    "CoreMLSessionError",
    "CoreMLSessionResult",
    "copy_session_options",
    "create_coreml_session",
    "default_coreml_cache_root",
]
