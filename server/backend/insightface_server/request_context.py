from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager


class RequestDeadlineExceeded(RuntimeError):
    """Raised in a worker before it commits work for an expired request."""


REQUEST_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "insightface_request_deadline", default=None
)


@contextmanager
def without_request_deadline() -> Iterator[None]:
    """Let an internal compensation restore an already-modified invariant."""

    token = REQUEST_DEADLINE.set(None)
    try:
        yield
    finally:
        REQUEST_DEADLINE.reset(token)


def remaining_seconds() -> float | None:
    deadline = REQUEST_DEADLINE.get()
    return None if deadline is None else deadline - time.monotonic()


def check_request_deadline() -> None:
    remaining = remaining_seconds()
    if remaining is not None and remaining <= 0:
        raise RequestDeadlineExceeded("The originating request deadline has expired")
