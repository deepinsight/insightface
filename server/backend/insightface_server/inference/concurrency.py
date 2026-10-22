from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ..request_context import check_request_deadline, remaining_seconds


class InferenceConcurrencyLimiter:
    """Bound all process-wide model work while allowing queued requests.

    The limiter is deliberately owned by the inference engine, so HTTP calls,
    enrollment work and RTSP frames all consume the same finite model budget.
    Closing rejects queued work and waits for active Session.run calls to drain
    before the model objects are released.
    """

    def __init__(self, max_concurrency: int) -> None:
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
            raise ValueError("inference max_concurrency must be an integer")
        if max_concurrency <= 0:
            raise ValueError("inference max_concurrency must be positive")
        self.max_concurrency = max_concurrency
        self._condition = threading.Condition()
        self._active = 0
        self._waiting = 0
        self._peak_active = 0
        self._closed = False

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._condition:
            if self._closed:
                raise RuntimeError("Inference engine is closed")
            self._waiting += 1
            try:
                while self._active >= self.max_concurrency:
                    check_request_deadline()
                    remaining = remaining_seconds()
                    timeout = None if remaining is None else min(0.1, max(0.001, remaining))
                    self._condition.wait(timeout=timeout)
                    if self._closed:
                        raise RuntimeError("Inference engine is closed")
                check_request_deadline()
                self._active += 1
                self._peak_active = max(self._peak_active, self._active)
            finally:
                self._waiting -= 1

        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close_and_wait(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._active:
                self._condition.wait()

    def summary(self) -> dict[str, int]:
        with self._condition:
            return {
                "max_concurrency": self.max_concurrency,
                "active": self._active,
                "waiting": self._waiting,
                "peak_active": self._peak_active,
            }
