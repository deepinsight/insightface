"""Single-source OpenCV demonstration of get -> match -> optional update."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import unquote, unquote_plus, urlsplit

import cv2

from ...app import PersonAnalysis
from ...app.person import PersonConfig
from .config import validate_person_config, validate_person_analysis_max_fps
from .model_packages import ensure_person_model, person_model_providers

_RTSP_PATTERN = re.compile(r'\b(?:rtsp|rtsps)://[^\s\"\'<>]+', re.IGNORECASE)


def display_time(timestamp, basis):
    if timestamp is None:
        return '—'
    if basis == 'utc':
        return datetime.fromtimestamp(timestamp / 1000, timezone.utc).astimezone().strftime('%H:%M:%S')
    seconds, milliseconds = divmod(max(0, int(timestamp)), 1000)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f'{hours:02d}:{minute:02d}:{second:02d}.{milliseconds:03d}'


@dataclass(frozen=True)
class PersonAnalysisJob:
    model_name: str
    model_root: Path
    provider_choice: str
    source_kind: str
    source: str | int
    references: tuple[tuple[str, Path], ...] = ()
    cache_dir: Path | None = None
    auto_update: bool = True
    person_config: dict = field(default_factory=dict)
    analysis_max_fps: float = 0.0

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError('A person model package is required')
        if not isinstance(self.auto_update, bool):
            raise ValueError('auto_update must be a boolean')
        object.__setattr__(self, 'person_config', validate_person_config(self.person_config))
        object.__setattr__(self, 'analysis_max_fps', validate_person_analysis_max_fps(self.analysis_max_fps))
        if self.source_kind == 'camera':
            if type(self.source) is not int or self.source < 0:
                raise ValueError('Camera source must be a nonnegative integer device index')
        elif self.source_kind == 'rtsp':
            if not isinstance(self.source, str):
                raise ValueError('RTSP source must be an rtsp/rtsps URL')
            parsed = urlsplit(self.source)
            if parsed.scheme.lower() not in ('rtsp', 'rtsps') or not parsed.hostname:
                raise ValueError('RTSP source must be an rtsp/rtsps URL')
        elif self.source_kind == 'video':
            if not isinstance(self.source, (str, Path)) or not str(self.source) or '://' in str(self.source):
                raise ValueError('Video source must be a local file path')
            object.__setattr__(self, 'source', str(Path(self.source).expanduser()))
        else:
            raise ValueError('Source kind must be video, camera, or rtsp')
        references = []
        for name, path in self.references:
            if not isinstance(name, str) or not name.strip():
                raise ValueError('Every reference photo requires a person name')
            references.append((name.strip(), Path(path).expanduser()))
        object.__setattr__(self, 'references', tuple(references))
        object.__setattr__(self, 'model_root', Path(self.model_root).expanduser())
        if self.cache_dir is not None:
            object.__setattr__(self, 'cache_dir', Path(self.cache_dir).expanduser())


@dataclass(frozen=True)
class FramePreview:
    matches: list
    frame_index: int
    timestamp: int | None
    time_basis: str


class _VideoClock:
    """Read media time, falling back to frame spacing when OpenCV lacks PTS."""
    def __init__(self, capture):
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.frame_ms = 1000. / fps if math.isfinite(fps) and fps > 0 else None
        self.last_ms = None

    def read(self, capture):
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
        if (not math.isfinite(timestamp) or timestamp < 0 or
                (self.last_ms is not None and timestamp <= self.last_ms)):
            if self.frame_ms is None:
                return None
            timestamp = 0. if self.last_ms is None else self.last_ms + self.frame_ms
        self.last_ms = timestamp
        return timestamp


class _LatestFrameReader:
    """Own live capture on one thread, retaining only the newest pending frame."""
    def __init__(self, capture):
        self._capture = capture
        self._condition = threading.Condition()
        self._latest = self._error = self._release_error = None
        self._stopped = self._done = False
        self._thread = threading.Thread(target=self._run, name='person-camera-capture', daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            while True:
                with self._condition:
                    if self._stopped:
                        break
                ok, frame = self._capture.read()
                read_at = time.time_ns() // 1_000_000
                if not ok:
                    raise RuntimeError('Camera stopped delivering frames. Check the connection and start again.')
                # A backend may reuse its buffer on the next read. The frame
                # being analyzed must remain independent of ongoing capture.
                sample = (frame.copy(), read_at)
                with self._condition:
                    if self._stopped:
                        break
                    self._latest = sample
                    self._condition.notify()
        except Exception as exc:
            self._error = exc
        finally:
            try:
                self._capture.release()
            except Exception as exc:
                self._release_error = exc
                self._error = self._error or exc
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def read(self, cancelled, not_before=0.):
        with self._condition:
            while True:
                if cancelled() or self._stopped:
                    return None
                delay = not_before - time.monotonic()
                if self._latest is not None and delay <= 0:
                    sample, self._latest = self._latest, None
                    return sample
                if self._latest is None and self._done:
                    if self._error is not None:
                        raise self._error
                    return None
                # Keep replacing the pending frame while waiting for the next
                # allowed start. Poll cancellation even if native read is blocked.
                self._condition.wait(min(delay, .05) if delay > 0 else .05)

    def close(self):
        with self._condition:
            self._stopped = True
            self._latest = None
            self._condition.notify_all()
        # Only the reader releases capture, after any native read returns.
        self._thread.join()
        if self._release_error is not None:
            raise self._release_error


class PersonAnalysisRunner:
    """Process videos in order and live inputs from their latest captured frame."""
    def __init__(self, job):
        if not isinstance(job, PersonAnalysisJob):
            raise TypeError('job must be PersonAnalysisJob')
        self.job = job
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._preview = None
        self._used = False
        self._state = dict(status='idle', frames_processed=0, registered=0,
                           body_references_added=0, body_references_replaced=0,
                           previews_replaced=0, stopped=False, error=None)

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _set(self, **values):
        with self._lock:
            self._state.update(values)

    def request_stop(self):
        # Never release a capture concurrently with its native read operation.
        self._cancel.set()
        with self._lock:
            if self._state['status'] in ('preparing', 'running'):
                self._state['status'] = 'stopping'

    def take_preview(self):
        with self._lock:
            value, self._preview = self._preview, None
            return value

    def _text(self, value):
        value = str(value)
        if self.job.source_kind == 'rtsp':
            value = value.replace(self.job.source, '[RTSP source]')
            parsed = urlsplit(self.job.source)
            secrets = set()
            for part in (parsed.username, parsed.password):
                if part:
                    secrets.update((part, unquote(part)))
            for pair in parsed.query.split('&'):
                key, _, part = pair.partition('=')
                if part and re.search(r'token|password|passwd|secret|auth|key|signature|credential', unquote_plus(key), re.I):
                    secrets.update((part, unquote_plus(part)))
            for secret in sorted(secrets, key=len, reverse=True):
                value = value.replace(secret, '[redacted]')
        return _RTSP_PATTERN.sub('[RTSP source]', value)

    def _open_capture(self):
        capture = cv2.VideoCapture()
        try:
            if self.job.source_kind != 'video':
                # Supported backends honor these open/read timeout requests.
                # Local device backends may reject them; RTSP never retries
                # without its deadlines. A blocked native call may still wait.
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
                network = self.job.source_kind == 'rtsp'
                backend = cv2.CAP_FFMPEG if network else cv2.CAP_ANY
                try:
                    opened = capture.open(self.job.source, backend, params)
                except (cv2.error, TypeError):
                    opened = False
                if not opened and not network:
                    capture.release()
                    capture = cv2.VideoCapture()
                    opened = capture.open(self.job.source, backend)
            else:
                opened = capture.open(self.job.source, cv2.CAP_ANY)
            if not opened or not capture.isOpened():
                raise RuntimeError('Unable to open the input. Check the file, device or RTSP connection.')
            if self.job.source_kind != 'video':
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend-dependent, best effort only.
            return capture
        except BaseException:
            capture.release()
            raise

    def run(self, progress=None, is_cancelled=lambda: False):
        with self._lock:
            if self._used:
                raise RuntimeError('Create a new PersonAnalysisRunner for each run')
            self._used = True
            self._state['status'] = 'preparing'
        def cancelled():
            return self._cancel.is_set() or bool(is_cancelled())
        def report(message, current=0, total=0):
            if progress is not None:
                progress(current, total, self._text(message))
        app = capture = reader = video_clock = error = None
        try:
            if cancelled():
                raise InterruptedError('Person analysis cancelled')
            report('Preparing person models…')
            ensure_person_model(self.job.model_name, self.job.model_root, self.job.cache_dir,
                                progress=progress, is_cancelled=cancelled)
            if cancelled():
                raise InterruptedError('Person analysis cancelled')
            app = PersonAnalysis(name=self.job.model_name, root=str(self.job.model_root),
                providers=person_model_providers(self.job.provider_choice),
                config=PersonConfig(**self.job.person_config))
            app.prepare()
            registered = set()
            for index, (name, path) in enumerate(self.job.references):
                if cancelled():
                    break
                report('Registering reference photos…', index, len(self.job.references))
                result = app.register(name, path)
                if result.accepted != 1 or result.rejected:
                    reasons = ', '.join(str(x.get('reason', 'unusable_face')) for x in result.rejected)
                    raise ValueError(f'Reference photo for {name!r} requires one usable face: {reasons}')
                registered.add(name)
                self._set(registered=len(registered))
            if not cancelled():
                capture = self._open_capture()
                if self.job.source_kind != 'video':
                    live = _LatestFrameReader(capture)
                    live.start()
                    reader, capture = live, None
                else:
                    video_clock = _VideoClock(capture)
                self._set(status='running')
            index, last_progress, next_analysis_at = 0, 0., 0.
            next_video_ms = 0.
            while (capture is not None or reader is not None) and not cancelled():
                if reader is not None:
                    sample = (reader.read(cancelled, not_before=next_analysis_at)
                              if self.job.analysis_max_fps else reader.read(cancelled))
                    if sample is None:
                        break
                    frame, read_at = sample
                else:
                    ok, frame = capture.read()
                    if not ok:
                        if index == 0 and not cancelled():
                            raise RuntimeError('No readable frames in this video. Check the file and video codec.')
                        break
                if cancelled():
                    break
                if video_clock is not None:
                    media_ms = video_clock.read(capture)
                    if self.job.analysis_max_fps:
                        if media_ms is None:
                            raise RuntimeError('Cannot limit video analysis: no usable video timestamps or FPS. '
                                               'Choose Auto or use a video with valid timing information.')
                        # Sample in media time without slowing file decoding to
                        # playback speed. Never run inference for skipped frames.
                        if media_ms + 1e-6 < next_video_ms:
                            continue
                        next_video_ms = media_ms + 1000. / self.job.analysis_max_fps
                if reader is not None and self.job.analysis_max_fps:
                    next_analysis_at = time.monotonic() + 1. / self.job.analysis_max_fps
                observations = app.get(frame)
                matches = app.match(observations)
                if self.job.auto_update:
                    update = app.update(matches)
                    with self._lock:
                        self._state['body_references_added'] += update.added
                        self._state['body_references_replaced'] += update.replaced
                index += 1
                if self.job.source_kind == 'video':
                    timestamp = round(media_ms) if media_ms is not None else None
                    basis = 'video'
                else:
                    timestamp, basis = read_at, 'utc'
                preview = (frame.copy(), FramePreview(deepcopy(matches), index, timestamp, basis))
                with self._lock:
                    self._state['previews_replaced'] += self._preview is not None
                    self._preview = preview
                    self._state['frames_processed'] = index
                if time.monotonic() - last_progress >= .2:
                    report('Analyzing person input…', index)
                    last_progress = time.monotonic()
        except Exception as exc:
            if not cancelled():
                error = exc
        finally:
            for resource in (reader, capture, app):
                if resource is not None:
                    try:
                        resource.release() if resource is capture else resource.close()
                    except Exception as exc:
                        error = error or exc
            self._set(stopped=cancelled(), status='failed' if error else 'stopped' if cancelled() else 'completed',
                      error=self._text(error) if error else None)
        if error is not None:
            raise RuntimeError(self._text(error)) from None
        return self.snapshot()
