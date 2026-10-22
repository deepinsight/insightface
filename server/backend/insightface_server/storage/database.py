from __future__ import annotations

import contextlib
import fcntl
import re
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from ..models.embedding_contract import legacy_embedding_contract_id
from ..request_context import check_request_deadline


class Database:
    _MIGRATION_NAME = re.compile(r"\d{4}_[A-Za-z0-9_.-]+\.sql")

    def __init__(self, path: Path, migrations_dir: Path):
        self.path = path
        self.migrations_dir = migrations_dir
        self._write_lock = threading.RLock()
        self._process_lock: BinaryIO | None = None

    def acquire_process_lock(self) -> None:
        """Reject a second Server process using the same durable data store."""

        with self._write_lock:
            if self._process_lock is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(self.path.suffix + ".server.lock")
            lock = lock_path.open("a+b")
            lock_path.chmod(0o600)
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock.close()
                raise RuntimeError(
                    "the data directory is already in use by another InsightFace "
                    "Server process; phase one requires exactly one worker"
                ) from exc
            self._process_lock = lock

    def release_process_lock(self) -> None:
        with self._write_lock:
            lock = self._process_lock
            self._process_lock = None
            if lock is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()

    def initialize(self) -> None:
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(self.path.suffix + ".migration.lock")
            with lock_path.open("a+b") as lock:
                lock_path.chmod(0o600)
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                with self._connect() as connection:
                    connection.create_function(
                        "legacy_embedding_contract_id", 5,
                        legacy_embedding_contract_id, deterministic=True,
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS schema_migrations "
                        "(version TEXT PRIMARY KEY, "
                        "applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                    )
                    connection.commit()
                    applied = {
                        row[0]
                        for row in connection.execute("SELECT version FROM schema_migrations")
                    }
                    candidates = (
                        path
                        for path in self.migrations_dir.glob("*.sql")
                        if self._MIGRATION_NAME.fullmatch(path.name)
                    )
                    for migration in sorted(candidates):
                        if migration.name in applied:
                            continue
                        # Migration names are constrained by _MIGRATION_NAME, so
                        # the quoted version cannot contain SQL metacharacters.
                        script = migration.read_text(encoding="utf-8")
                        transaction = (
                            "BEGIN IMMEDIATE;\n"
                            + script
                            + "\nINSERT INTO schema_migrations(version) VALUES "
                            + f"('{migration.name}');\nCOMMIT;"
                        )
                        try:
                            connection.executescript(transaction)
                        except Exception:
                            connection.rollback()
                            raise
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        self.path.chmod(0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextlib.contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            check_request_deadline()
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                check_request_deadline()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def status(self) -> dict[str, object]:
        with self.read() as connection:
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
            count = connection.execute(
                "SELECT count(*) FROM schema_migrations"
            ).fetchone()[0]
        return {
            "path": str(self.path),
            "quick_check": quick,
            "migration_count": int(count),
            "wal": True,
        }
