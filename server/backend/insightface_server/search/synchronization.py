from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class ReadWriteLock:
    """Small writer-preferring lock for immutable index-generation reads."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._writer = False
        self._writer_owner: int | None = None
        self._writer_depth = 0
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            existing_depth = self._reader_depths.get(owner, 0)
            while (
                self._writer_owner not in {None, owner}
                or (self._waiting_writers and existing_depth == 0 and not self._writer)
            ):
                self._condition.wait()
            self._readers += 1
            self._reader_depths[owner] = existing_depth + 1

    def release_read(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            depth = self._reader_depths.get(owner, 0)
            if self._readers <= 0 or depth <= 0:
                raise RuntimeError("read lock released without being acquired")
            self._readers -= 1
            if depth == 1:
                self._reader_depths.pop(owner)
            else:
                self._reader_depths[owner] = depth - 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_owner == owner:
                self._writer_depth += 1
                return
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
                self._writer_owner = owner
                self._writer_depth = 1
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            if not self._writer or self._writer_owner != owner:
                raise RuntimeError("write lock released without being acquired")
            self._writer_depth -= 1
            if self._writer_depth:
                return
            self._writer = False
            self._writer_owner = None
            self._condition.notify_all()

    @contextmanager
    def read(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
