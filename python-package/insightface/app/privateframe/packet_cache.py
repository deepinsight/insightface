"""SQLite-backed bounded cache of encoded video packets and GOP metadata."""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class CachedVideoInfo:
    codec_name: str
    extradata: bytes
    rotation_degrees: int


def _rotation(path: str | Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not read orientation metadata: {path}")
    try:
        value = round(capture.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    finally:
        capture.release()
    if value not in {0, 90, 180, 270}:
        raise ValueError(f"unsupported video orientation: {value}")
    return value


def orient_bgr(value: np.ndarray, rotation_degrees: int) -> np.ndarray:
    if rotation_degrees == 0:
        return value
    if rotation_degrees == 90:
        return cv2.rotate(value, cv2.ROTATE_90_CLOCKWISE)
    if rotation_degrees == 180:
        return cv2.rotate(value, cv2.ROTATE_180)
    if rotation_degrees == 270:
        return cv2.rotate(value, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {rotation_degrees}")


def crop_bgr(
    image: np.ndarray, crop: tuple[int, int, int, int]
) -> np.ndarray:
    """Return one exact, zero-padded source-coordinate BGR rectangle."""

    x1, y1, x2, y2 = crop
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid crop rectangle: {crop}")
    canvas = np.zeros((y2 - y1, x2 - x1, 3), dtype=image.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = image[
            sy1:sy2, sx1:sx2
        ]
    return canvas


class DecodedFrameStore:
    """Byte-bounded LRU of complete, display-oriented BGR frames.

    Historical consumers ask for their own crops, but the store always loads
    and retains the complete frame.  A later request for another crop can then
    reuse the same decode.  Missing contiguous ranges are split before calling
    ``loader`` so one decode result does not exceed the store's intended memory
    budget on fixed-resolution video.

    ``frame_target`` and ``byte_capacity`` are both cache limits.  Setting
    either one to zero disables retention while preserving range-read
    semantics.
    """

    def __init__(self, frame_target: int, byte_capacity: int):
        self.frame_target = int(frame_target)
        self.byte_capacity = int(byte_capacity)
        if self.frame_target < 0:
            raise ValueError("frame_target cannot be negative")
        if self.byte_capacity < 0:
            raise ValueError("byte_capacity cannot be negative")
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._live_bytes = 0
        self._peak_bytes = 0
        self._estimated_frame_bytes = 0
        self.hits = 0

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def live_bytes(self) -> int:
        return self._live_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    def remember(self, frame_index: int, frame: np.ndarray) -> bool:
        """Remember one complete BGR frame, returning whether it was retained."""

        self._validate_frame(frame)
        frame_index = int(frame_index)
        frame_bytes = int(frame.nbytes)
        self._estimated_frame_bytes = max(self._estimated_frame_bytes, frame_bytes)

        previous = self._frames.pop(frame_index, None)
        if previous is not None:
            self._live_bytes -= int(previous.nbytes)

        if (
            self.frame_target == 0
            or self.byte_capacity == 0
            or frame_bytes > self.byte_capacity
        ):
            return False

        self._frames[frame_index] = frame
        self._live_bytes += frame_bytes
        while self._frames and (
            len(self._frames) > self.frame_target
            or self._live_bytes > self.byte_capacity
        ):
            _oldest_index, oldest = self._frames.popitem(last=False)
            self._live_bytes -= int(oldest.nbytes)
        self._peak_bytes = max(self._peak_bytes, self._live_bytes)
        return frame_index in self._frames

    def read_range(
        self,
        first_frame: int,
        last_frame: int,
        *,
        loader: Callable[[int, int], Mapping[int, np.ndarray]],
        crop: tuple[int, int, int, int] | None = None,
        crops: Mapping[int, tuple[int, int, int, int]] | None = None,
    ) -> dict[int, np.ndarray]:
        """Read a frame interval, decoding only uncached contiguous portions.

        ``loader`` must return complete, already-oriented BGR frames.  Cropping
        is deliberately applied only after each complete frame is remembered.
        """

        first_frame = int(first_frame)
        last_frame = int(last_frame)
        if first_frame > last_frame:
            return {}
        if crop is not None and crops is not None:
            raise ValueError("crop and crops are mutually exclusive")

        output: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for frame_index in range(first_frame, last_frame + 1):
            frame = self._frames.pop(frame_index, None)
            if frame is None:
                missing.append(frame_index)
                continue
            self._frames[frame_index] = frame
            self.hits += 1
            output[frame_index] = self._select(frame, frame_index, crop, crops)

        for interval_first, interval_last in self._missing_intervals(missing):
            chunk_first = interval_first
            while chunk_first <= interval_last:
                chunk_size = self._decode_chunk_size(interval_last - chunk_first + 1)
                chunk_last = min(interval_last, chunk_first + chunk_size - 1)
                self._reserve_decode_block(chunk_last - chunk_first + 1)
                loaded = loader(chunk_first, chunk_last)
                absent = [
                    frame_index
                    for frame_index in range(chunk_first, chunk_last + 1)
                    if frame_index not in loaded
                ]
                if absent:
                    raise RuntimeError(
                        f"decoded frame loader omitted frames: {absent[:8]}"
                    )
                for frame_index in range(chunk_first, chunk_last + 1):
                    frame = loaded[frame_index]
                    self.remember(frame_index, frame)
                    output[frame_index] = self._select(
                        frame, frame_index, crop, crops
                    )
                chunk_first = chunk_last + 1

        return {
            frame_index: output[frame_index]
            for frame_index in range(first_frame, last_frame + 1)
        }

    def clear(self) -> None:
        self._frames.clear()
        self._live_bytes = 0

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("decoded frame must be an HxWx3 BGR numpy array")

    @staticmethod
    def _missing_intervals(missing: list[int]) -> Iterator[tuple[int, int]]:
        if not missing:
            return
        first = last = missing[0]
        for frame_index in missing[1:]:
            if frame_index == last + 1:
                last = frame_index
                continue
            yield first, last
            first = last = frame_index
        yield first, last

    def _decode_chunk_size(self, remaining: int) -> int:
        if self.frame_target == 0 or self.byte_capacity == 0:
            return remaining
        if self._estimated_frame_bytes <= 0:
            return 1
        frames_by_bytes = max(
            1, self.byte_capacity // self._estimated_frame_bytes
        )
        return max(1, min(remaining, self.frame_target, frames_by_bytes))

    def _reserve_decode_block(self, frame_count: int) -> None:
        """Evict before a loader materializes its fixed-resolution block."""

        if (
            self.byte_capacity == 0
            or self._estimated_frame_bytes <= 0
            or frame_count <= 0
        ):
            return
        block_bytes = min(
            self.byte_capacity,
            int(frame_count) * self._estimated_frame_bytes,
        )
        maximum_live_bytes = self.byte_capacity - block_bytes
        while self._frames and self._live_bytes > maximum_live_bytes:
            _oldest_index, oldest = self._frames.popitem(last=False)
            self._live_bytes -= int(oldest.nbytes)

    @staticmethod
    def _select(
        frame: np.ndarray,
        frame_index: int,
        crop: tuple[int, int, int, int] | None,
        crops: Mapping[int, tuple[int, int, int, int]] | None,
    ) -> np.ndarray:
        selected_crop = crops[frame_index] if crops is not None else crop
        return crop_bgr(frame, selected_crop) if selected_crop is not None else frame


class EncodedPacketCache:
    """Append packets once, re-decode bounded historical intervals on demand.

    Rows are deleted only at a random-access boundary. SQLite may retain the
    allocated pages for reuse, but the live packet payload remains bounded.
    """

    def __init__(self, path: str | Path, source: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self._closed = False
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS packets (
                    sequence INTEGER PRIMARY KEY,
                    pts INTEGER,
                    dts INTEGER,
                    duration INTEGER,
                    time_base_num INTEGER NOT NULL,
                    time_base_den INTEGER NOT NULL,
                    is_keyframe INTEGER NOT NULL,
                    gop_sequence INTEGER NOT NULL,
                    payload BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS packets_gop ON packets(gop_sequence, sequence);
                CREATE INDEX IF NOT EXISTS packets_pts ON packets(pts, sequence);
                CREATE TABLE IF NOT EXISTS frames (
                    frame_index INTEGER PRIMARY KEY,
                    pts INTEGER,
                    packet_sequence INTEGER NOT NULL,
                    gop_sequence INTEGER NOT NULL
                );
                """
            )
            self.rotation_degrees = _rotation(source)
        except BaseException:
            self._closed = True
            try:
                self.connection.close()
            finally:
                self._remove_runtime_files()
            raise
        self.sequence = -1
        self.gop_sequence = 0
        self.last_packet_sequence = -1
        self._pending_since_commit = 0
        self.info: CachedVideoInfo | None = None
        self.frames_decoded = 0
        self.evicted_packets = 0
        self.evicted_bytes = 0
        self.historical_decode_requests = 0
        self.historical_packets_read = 0
        self.peak_decode_range_bytes = 0

    def configure(self, stream: Any) -> None:
        self.info = CachedVideoInfo(
            codec_name=str(stream.codec_context.name),
            extradata=bytes(stream.codec_context.extradata or b""),
            rotation_degrees=self.rotation_degrees,
        )

    def append_packet(self, packet: Any) -> tuple[int, int] | None:
        payload = bytes(packet)
        if not payload:
            return None
        self.sequence += 1
        if bool(packet.is_keyframe):
            self.gop_sequence = self.sequence
        time_base = Fraction(packet.time_base)
        self.connection.execute(
            "INSERT INTO packets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.sequence,
                packet.pts,
                packet.dts,
                packet.duration,
                time_base.numerator,
                time_base.denominator,
                int(bool(packet.is_keyframe)),
                self.gop_sequence,
                sqlite3.Binary(payload),
            ),
        )
        self.last_packet_sequence = self.sequence
        self._pending_since_commit += 1
        if self._pending_since_commit >= 120:
            self.connection.commit()
            self._pending_since_commit = 0
        return self.sequence, self.gop_sequence

    def record_frame(self, frame_index: int, pts: int | None, packet_sequence: int, gop_sequence: int) -> None:
        if pts is None:
            raise RuntimeError(
                "video frame has no PTS; reliable GOP history decoding is unsupported"
            )
        if pts is not None:
            owner = self.connection.execute(
                "SELECT sequence, gop_sequence FROM packets WHERE pts = ? ORDER BY sequence LIMIT 1",
                (pts,),
            ).fetchone()
            if owner is not None:
                packet_sequence, gop_sequence = int(owner[0]), int(owner[1])
        self.connection.execute(
            "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?)",
            (frame_index, pts, packet_sequence, gop_sequence),
        )
        self.frames_decoded = max(self.frames_decoded, frame_index + 1)

    def commit(self) -> None:
        self.connection.commit()
        self._pending_since_commit = 0

    def evict_before_frame(self, frame_index: int) -> None:
        row = self.connection.execute(
            "SELECT gop_sequence FROM frames WHERE frame_index >= ? ORDER BY frame_index LIMIT 1",
            (max(0, frame_index),),
        ).fetchone()
        if row is None:
            return
        keep_gop = int(row[0])
        aggregate = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM packets WHERE sequence < ?",
            (keep_gop,),
        ).fetchone()
        self.evicted_packets += int(aggregate[0])
        self.evicted_bytes += int(aggregate[1])
        self.connection.execute("DELETE FROM packets WHERE sequence < ?", (keep_gop,))
        self.connection.execute("DELETE FROM frames WHERE gop_sequence < ?", (keep_gop,))
        self.connection.commit()

    def decode_range(
        self,
        first_frame: int,
        last_frame: int,
        *,
        crop: tuple[int, int, int, int] | None = None,
        crops: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> dict[int, np.ndarray]:
        if first_frame > last_frame:
            return {}
        if crop is not None and crops is not None:
            raise ValueError("crop and crops are mutually exclusive")
        if self.info is None:
            raise RuntimeError("packet cache has not been configured")
        start = self.connection.execute(
            "SELECT gop_sequence FROM frames WHERE frame_index <= ? ORDER BY frame_index DESC LIMIT 1",
            (first_frame,),
        ).fetchone()
        end = self.connection.execute(
            "SELECT packet_sequence FROM frames WHERE frame_index >= ? ORDER BY frame_index LIMIT 1",
            (last_frame,),
        ).fetchone()
        if start is None or end is None:
            raise KeyError(f"frames {first_frame}..{last_frame} are outside the encoded cache")
        rows = self.connection.execute(
            "SELECT sequence, pts, dts, duration, time_base_num, time_base_den, payload "
            "FROM packets WHERE sequence >= ? AND sequence <= ? ORDER BY sequence",
            (int(start[0]), self.last_packet_sequence),
        )
        wanted_pts = {
            int(row[1]): int(row[0])
            for row in self.connection.execute(
                "SELECT frame_index, pts FROM frames WHERE frame_index BETWEEN ? AND ?",
                (first_frame, last_frame),
            )
            if row[1] is not None
        }
        decoder = av.CodecContext.create(self.info.codec_name, "r")
        decoder.extradata = self.info.extradata
        output: dict[int, np.ndarray] = {}

        def receive(frames: list[Any]) -> None:
            for frame in frames:
                if frame.pts is None or int(frame.pts) not in wanted_pts:
                    continue
                index = wanted_pts[int(frame.pts)]
                image = orient_bgr(
                    frame.to_ndarray(format="bgr24"), self.info.rotation_degrees
                )
                selected_crop = crops[index] if crops is not None else crop
                if selected_crop is not None:
                    image = crop_bgr(image, selected_crop)
                output[index] = image

        self.historical_decode_requests += 1
        try:
            for _sequence, pts, dts, duration, num, den, payload in rows:
                self.historical_packets_read += 1
                packet = av.Packet(bytes(payload))
                packet.pts = pts
                packet.dts = dts
                packet.duration = duration
                packet.time_base = Fraction(int(num), int(den))
                receive(decoder.decode(packet))
                if len(output) == len(wanted_pts):
                    break
        finally:
            rows.close()
        if len(output) != len(wanted_pts):
            receive(decoder.decode(None))
        missing = sorted(set(range(first_frame, last_frame + 1)) - set(output))
        if missing:
            raise RuntimeError(f"packet cache failed to decode frames: {missing[:8]}")
        self.peak_decode_range_bytes = max(
            self.peak_decode_range_bytes,
            sum(int(image.nbytes) for image in output.values()),
        )
        return output

    def oldest_frame_index(self) -> int | None:
        """Return the oldest frame whose random-access GOP is still cached."""

        row = self.connection.execute(
            "SELECT MIN(frame_index) FROM frames"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def live_payload_bytes(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(LENGTH(payload)), 0) FROM packets"
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        if self._closed:
            self._remove_runtime_files()
            return
        self._closed = True
        try:
            self.connection.commit()
            self._pending_since_commit = 0
        finally:
            try:
                self.connection.close()
            finally:
                self._remove_runtime_files()

    def _remove_runtime_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                self.path.with_name(f"{self.path.name}{suffix}").unlink()
            except FileNotFoundError:
                pass


def iter_cached_frames(
    source: str | Path, cache: EncodedPacketCache
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Demux once, append encoded packets, and yield display-oriented BGR frames."""

    container = av.open(str(source))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    cache.configure(stream)
    average_rate = float(stream.average_rate or 0.0)
    frame_index = 0
    try:
        for packet in container.demux(stream):
            marker = cache.append_packet(packet)
            frames = packet.decode()
            for frame in frames:
                if marker is None:
                    marker = (cache.last_packet_sequence, cache.gop_sequence)
                cache.record_frame(frame_index, frame.pts, marker[0], marker[1])
                timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else (
                    frame_index / average_rate if average_rate > 0.0 else 0.0
                )
                yield frame_index, timestamp, orient_bgr(
                    frame.to_ndarray(format="bgr24"), cache.rotation_degrees
                )
                frame_index += 1
    finally:
        try:
            cache.commit()
        finally:
            container.close()


def iter_oriented_frames(
    source: str | Path,
) -> Iterator[tuple[int, float, int | None, np.ndarray]]:
    """Decode one display-oriented pass without creating a packet cache."""

    container = av.open(str(source))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rotation_degrees = _rotation(source)
    average_rate = float(stream.average_rate or 0.0)
    try:
        for frame_index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else frame_index / average_rate
                if average_rate > 0.0
                else 0.0
            )
            yield (
                frame_index,
                timestamp,
                int(frame.pts) if frame.pts is not None else None,
                orient_bgr(
                    frame.to_ndarray(format="bgr24"), rotation_degrees
                ),
            )
    finally:
        container.close()


__all__ = [
    "DecodedFrameStore",
    "EncodedPacketCache",
    "crop_bgr",
    "iter_cached_frames",
    "iter_oriented_frames",
    "orient_bgr",
]
