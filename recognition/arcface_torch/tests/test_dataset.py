"""DataLoaderX / BackgroundGenerator behaviour (dataset.py)."""
import sys
sys.path.insert(0, ".")
import threading
import pytest


def _run_with_timeout(fn, timeout=5.0):
    out = {}
    def target():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e
    t = threading.Thread(target=target, daemon=True)
    t.start(); t.join(timeout)
    assert not t.is_alive(), "consumer hung instead of raising"
    return out


def test_background_generator_propagates_worker_exception():
    """An exception in the prefetch thread must surface in the consumer, not deadlock it
    (a hung rank turns into a 10-minute NCCL timeout on every other rank)."""
    from dataset import BackgroundGenerator

    def gen():
        yield 1
        raise RuntimeError("pin_memory blew up")

    bg = BackgroundGenerator(gen(), local_rank=None)
    assert _run_with_timeout(lambda: next(bg)).get("value") == 1
    out = _run_with_timeout(lambda: next(bg))
    assert isinstance(out.get("error"), RuntimeError), out
    assert "pin_memory blew up" in str(out["error"])


def test_background_generator_ends_cleanly():
    from dataset import BackgroundGenerator
    bg = BackgroundGenerator(iter([1, 2]), local_rank=None)
    assert [_run_with_timeout(lambda: next(bg)).get("value") for _ in range(2)] == [1, 2]
    out = _run_with_timeout(lambda: next(bg))
    assert isinstance(out.get("error"), StopIteration), out
