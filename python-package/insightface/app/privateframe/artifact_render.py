"""Render debug or redacted MP4 files from finalized observation artifacts."""

from __future__ import annotations

import gc
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import av
import cv2
import numpy as np

from .artifacts import sha256_file
from .geometry import clip
from .packet_cache import iter_oriented_frames
from .recognition import apply_identity_policy, _reference_file_name
from .video import paths_are_distinct, probe_video, temporary_video_path


@dataclass(frozen=True)
class RenderTarget:
    mode: str
    path: Path


class _VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def finish(self) -> None: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


def _release_native_writer_cycles(
    writers: list[_VideoWriter],
    *,
    pyav: bool,
) -> None:
    if not pyav:
        return
    writers.clear()
    gc.collect()


def _abort_writers(writers: list[_VideoWriter], *, pyav: bool) -> None:
    """Clean every writer without replacing the error that caused the abort."""

    for index in range(len(writers)):
        with suppress(BaseException):
            writers[index].abort()
    with suppress(BaseException):
        _release_native_writer_cycles(writers, pyav=pyav)


def _raise_if_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise InterruptedError("PrivateFrame operation was cancelled")


def _color(track_id: str) -> tuple[int, int, int]:
    value = (int(track_id[1:]) + 1) * 2654435761 & 0xFFFFFFFF
    return (
        80 + (value & 127),
        80 + ((value >> 8) & 127),
        80 + ((value >> 16) & 127),
    )


def _dashed_line(
    frame: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int,
    gap: int,
) -> None:
    length = round(np.linalg.norm(np.asarray(second) - np.asarray(first)))
    if length <= 0:
        return
    direction = (np.asarray(second, dtype=float) - np.asarray(first, dtype=float)) / length
    for start in range(0, length + 1, dash + gap):
        end = min(length, start + dash)
        a = np.asarray(first) + direction * start
        b = np.asarray(first) + direction * end
        cv2.line(
            frame,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _debug_box(
    frame: np.ndarray,
    item: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    value = clip(item["box"], frame.shape[1] - 1, frame.shape[0] - 1)
    x1, y1, x2, y2 = np.rint(value).astype(int)
    color = _color(item["track_id"])
    thickness = int(settings.get("debug_line_thickness", 2))
    if item["source"] == "detector":
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA,
        )
    else:
        dash, gap = (11, 5) if int(item.get("local_match_count", 0)) >= 1 else (2, 5)
        _dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash, gap)
        _dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash, gap)
        _dashed_line(frame, (x2, y2), (x1, y2), color, thickness, dash, gap)
        _dashed_line(frame, (x1, y2), (x1, y1), color, thickness, dash, gap)
    cv2.putText(
        frame,
        item["track_id"],
        (x1, max(15, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _debug_frame_number(frame: np.ndarray, frame_idx: int) -> None:
    """Draw a prominent frame number away from video-player controls."""

    label = f"FRAME {frame_idx:06d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.65, min(1.2, min(frame.shape[:2]) / 1080.0 * 0.9))
    thickness = 2
    padding = 12
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    panel_width = text_width + padding * 2
    panel_height = text_height + baseline + padding * 2
    cv2.rectangle(
        frame,
        (0, 0),
        (panel_width, panel_height),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (padding, padding + text_height),
        font,
        scale,
        (80, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _odd_kernel(value: float) -> int:
    kernel = max(1, round(value))
    return kernel if kernel % 2 == 1 else kernel + 1


def _gaussian_blur(region: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    kernel = _odd_kernel(
        max(
            int(settings["min_kernel"]),
            round(max(region.shape[:2]) * float(settings["kernel_ratio"])),
        )
    )
    sigma = float(settings.get("sigma", 0.0))
    algorithm = str(settings.get("algorithm", "exact"))
    max_side = int(settings.get("max_side", 64))

    if algorithm == "pyramid" and max(region.shape[:2]) > max_side:
        scale = max_side / max(region.shape[:2])
        reduced_width = max(1, round(region.shape[1] * scale))
        reduced_height = max(1, round(region.shape[0] * scale))
        reduced = cv2.resize(
            region,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )
        reduced_kernel = _odd_kernel(kernel * scale)
        reduced_sigma = sigma * scale if sigma > 0.0 else 0.0
        reduced = cv2.GaussianBlur(
            reduced,
            (reduced_kernel, reduced_kernel),
            sigmaX=reduced_sigma,
            sigmaY=reduced_sigma,
        )
        return cv2.resize(
            reduced,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.GaussianBlur(
        region,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def _blur(frame: np.ndarray, value: list[float], settings: dict[str, Any]) -> None:
    redaction = settings["redaction"]
    method = str(redaction["method"])
    scale = float(redaction["box_scale"])
    gaussian = redaction.get("gaussian", {})
    mosaic = redaction.get("mosaic", {})
    feather = redaction.get("feather", {"enabled": False})

    target = np.asarray(value, dtype=float)
    width, height = target[2] - target[0], target[3] - target[1]
    center_x = (target[0] + target[2]) * 0.5
    center_y = (target[1] + target[3]) * 0.5
    target = np.asarray(
        [
            center_x - width * scale * 0.5,
            center_y - height * scale * 0.5,
            center_x + width * scale * 0.5,
            center_y + height * scale * 0.5,
        ]
    )
    x1, y1, x2, y2 = np.rint(clip(target, frame.shape[1], frame.shape[0])).astype(int)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if method == "gaussian":
        redacted = _gaussian_blur(region, gaussian)
    elif method == "mosaic":
        block = max(
            int(mosaic["min_block_size"]),
            round(max(region.shape[:2]) * float(mosaic["block_size_ratio"])),
        )
        reduced_width = max(1, (region.shape[1] + block - 1) // block)
        reduced_height = max(1, (region.shape[0] + block - 1) // block)
        reduced = cv2.resize(
            region,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )
        redacted = cv2.resize(
            reduced,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:  # Defensive guard for callers that bypass configuration loading.
        raise ValueError(f"unsupported redaction method: {method}")

    if not bool(feather.get("enabled", False)):
        frame[y1:y2, x1:x2] = redacted
        return
    feather_pixels = max(
        int(feather.get("min_pixels", 1)),
        round(min(region.shape[:2]) * float(feather.get("ratio", 0.10))),
    )
    row_distance = np.minimum(
        np.arange(region.shape[0]),
        np.arange(region.shape[0] - 1, -1, -1),
    )[:, None]
    column_distance = np.minimum(
        np.arange(region.shape[1]),
        np.arange(region.shape[1] - 1, -1, -1),
    )[None, :]
    alpha = np.clip(
        np.minimum(row_distance, column_distance) / max(feather_pixels, 1),
        0.0,
        1.0,
    )
    # Smoothstep avoids a visible linear seam around the redaction region.
    alpha = (alpha * alpha * (3.0 - 2.0 * alpha))[..., None]
    blended = redacted.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
    frame[y1:y2, x1:x2] = np.rint(blended).astype(np.uint8)


def _identity_should_blur(
    item: dict[str, Any],
    policy: dict[str, Any],
    recognition: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Apply reference membership and the selected unconfirmed-face action."""

    mode = str(policy.get("mode", "all"))
    action = policy.get("unknown_action", "auto")
    if mode == "all":
        decision = apply_identity_policy(mode, None, action)
        return decision.should_blur, decision.reason
    if not isinstance(recognition, dict) or recognition.get("enabled") is not True:
        raise ValueError("selective rendering requires recognition results")
    references = recognition.get("references")
    files = references.get("files") if isinstance(references, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("selective rendering requires usable reference photos")
    known_files = set()
    for value in files:
        if not isinstance(value, dict) or not isinstance(value.get("file"), str) or not value["file"]:
            raise ValueError("invalid reference photo record in analysis result")
        known_files.add(_reference_file_name(value["file"]))
    tracks = recognition.get("tracks")
    record = tracks.get(str(item.get("track_id"))) if isinstance(tracks, dict) else None
    if not isinstance(record, dict):
        raise ValueError("missing reference-match decision for rendered track")
    # Validate the stored match before considering any per-observation override.
    decision = apply_identity_policy(mode, record, action)
    matched_files = {_reference_file_name(name) for name in record.get("matched_reference_files", [])}
    if not matched_files <= known_files:
        raise ValueError("track match names a photo absent from the analyzed reference set")
    if item.get("identity_unconfirmed") is True or (
        item.get("endpoint_repair_reason") == "interpolate_unanchored_endpoint"
    ):
        # Endpoint geometry alone does not establish identity continuity. Both
        # public JSON and in-memory artifacts use the same unknown-face policy.
        decision = apply_identity_policy(
            mode,
            {"status": "UNKNOWN", "matched_reference_files": [],
             "reason": "unconfirmed_interpolate_endpoint"},
            action,
        )
    return bool(decision.should_blur), str(decision.reason)


def _bitrate(value: Any) -> int:
    text = str(value).strip().lower()
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    if text and text[-1] in multipliers:
        return round(float(text[:-1]) * multipliers[text[-1]])
    return int(text)


def _stream_from_template(container: Any, stream: Any) -> Any:
    method = getattr(container, "add_stream_from_template", None)
    if method is not None:
        return method(stream)
    return container.add_stream(template=stream)


def _video_arguments(settings: dict[str, Any]) -> list[str]:
    encoder = str(settings["encoder"])
    arguments = ["-c:v", encoder]
    preset = settings.get("preset")
    if preset:
        arguments.extend(["-preset", str(preset)])
    rate = settings["rate_control"]
    mode = str(rate["mode"])
    if mode in {"crf", "cq"}:
        quality = str(int(rate["quality"]))
        if "nvenc" in encoder:
            arguments.extend(["-rc:v", "vbr", "-cq:v", quality, "-b:v", "0"])
        else:
            arguments.extend(["-crf", quality])
    elif mode == "vbr":
        arguments.extend(["-b:v", str(rate["bitrate"])])
        if rate.get("max_bitrate"):
            arguments.extend(["-maxrate", str(rate["max_bitrate"])])
    elif mode == "cbr":
        bitrate = str(rate["bitrate"])
        arguments.extend(["-b:v", bitrate, "-minrate", bitrate, "-maxrate", bitrate])
        if rate.get("buffer_size"):
            arguments.extend(["-bufsize", str(rate["buffer_size"])])
    arguments.extend(["-pix_fmt", str(settings["pixel_format"])])
    if int(settings.get("keyframe_interval", 0)) > 0:
        arguments.extend(["-g", str(int(settings["keyframe_interval"]))])
    if bool(settings.get("faststart", True)):
        arguments.extend(["-movflags", "+faststart"])
    return arguments


def _pyav_video_options(settings: dict[str, Any]) -> dict[str, str]:
    encoder = str(settings["encoder"])
    options: dict[str, str] = {}
    if settings.get("preset"):
        options["preset"] = str(settings["preset"])
    rate = settings["rate_control"]
    mode = str(rate["mode"])
    if mode in {"crf", "cq"}:
        quality = str(int(rate["quality"]))
        if "nvenc" in encoder:
            options.update({"rc": "vbr", "cq": quality, "b": "0"})
        else:
            options["crf"] = quality
    elif mode == "cbr":
        options["minrate"] = str(rate["bitrate"])
        options["maxrate"] = str(rate["bitrate"])
        if rate.get("buffer_size"):
            options["bufsize"] = str(rate["buffer_size"])
    elif rate.get("max_bitrate"):
        options["maxrate"] = str(rate["max_bitrate"])
    return options


def _copy_audio(
    silent_video: Path,
    source: Path,
    destination: Path,
    *,
    requested_mode: str,
    maximum_duration: float,
) -> bool:
    """Remux source audio with the encoded video; return false if absent."""

    containers = []
    try:
        video_input = av.open(str(silent_video))
        containers.append(video_input)
        audio_input = av.open(str(source))
        containers.append(audio_input)
        audio_stream = next(iter(audio_input.streams.audio), None)
        if audio_stream is None:
            return False
        if requested_mode == "aac" and audio_stream.codec_context.name != "aac":
            raise RuntimeError(
                "PyAV audio mode aac currently requires AAC source audio; "
                "choose copy/none or use the ffmpeg backend for transcoding"
            )
        output = av.open(str(destination), "w", options={"movflags": "+faststart"})
        containers.append(output)
        output_video = _stream_from_template(output, video_input.streams.video[0])
        output_audio = _stream_from_template(output, audio_stream)
        for packet in video_input.demux(video_input.streams.video[0]):
            if packet.dts is None:
                continue
            packet.stream = output_video
            output.mux(packet)
        for packet in audio_input.demux(audio_stream):
            if packet.dts is None:
                continue
            timestamp = (
                float(packet.pts * packet.time_base)
                if packet.pts is not None and packet.time_base is not None
                else 0.0
            )
            if timestamp >= maximum_duration:
                break
            packet.stream = output_audio
            output.mux(packet)
    finally:
        propagating_error = sys.exc_info()[0] is not None
        close_error = None
        for container in reversed(containers):
            try:
                container.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if close_error is not None and not propagating_error:
            raise close_error
    return True


class _PyAVWriter:
    def __init__(
        self,
        destination: Path,
        source: Path,
        width: int,
        height: int,
        fps: float,
        settings: dict[str, Any],
        audio_mode: str,
    ):
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = temporary_video_path(destination)
        self.muxed: Path | None = None
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = 0
        self.audio_mode = audio_mode
        self.closed = True
        try:
            self.container = av.open(
                str(self.temporary),
                "w",
                options={"movflags": "+faststart"} if bool(settings.get("faststart", True)) else None,
            )
            self.closed = False
            rate = Fraction(str(fps)).limit_denominator(1_000_000)
            self.stream = self.container.add_stream(str(settings["encoder"]), rate=rate)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = str(settings["pixel_format"])
            self.stream.options = _pyav_video_options(settings)
            rate_control = settings["rate_control"]
            if str(rate_control["mode"]) in {"vbr", "cbr"}:
                self.stream.bit_rate = _bitrate(rate_control["bitrate"])
            if int(settings.get("keyframe_interval", 0)) > 0:
                self.stream.codec_context.gop_size = int(settings["keyframe_interval"])
        except BaseException:
            # The caller cannot register this writer until construction succeeds.
            with suppress(BaseException):
                self.abort()
            raise

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"renderer frame shape changed: {frame.shape} != {(self.height, self.width, 3)}")
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)
        self.frames += 1

    def finish(self) -> None:
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.closed = True
        if self.audio_mode == "none":
            return
        self.muxed = temporary_video_path(self.destination)
        if not _copy_audio(
            self.temporary,
            self.source,
            self.muxed,
            requested_mode=self.audio_mode,
            maximum_duration=self.frames / self.fps,
        ):
            self.muxed = None

    def commit(self) -> None:
        ready = self.muxed or self.temporary
        os.replace(ready, self.destination)
        if ready != self.temporary and self.temporary.exists():
            self.temporary.unlink()

    def abort(self) -> None:
        if not self.closed:
            with suppress(BaseException):
                self.container.close()
            self.closed = True
        for path in (self.temporary, self.muxed):
            if path is not None and path.exists():
                path.unlink()


class _FFmpegWriter:
    def __init__(
        self,
        destination: Path,
        source: Path,
        width: int,
        height: int,
        fps: float,
        settings: dict[str, Any],
        audio_mode: str,
    ):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required for artifact video rendering")
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = temporary_video_path(destination)
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
        ]
        if audio_mode != "none":
            command.extend(["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?"])
        else:
            command.extend(["-map", "0:v:0"])
        command.extend(_video_arguments(settings))
        if audio_mode == "copy":
            command.extend(["-c:a", "copy", "-shortest"])
        elif audio_mode == "aac":
            command.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    str(settings.get("audio", {}).get("bitrate", "192k")),
                    "-shortest",
                ]
            )
        else:
            command.append("-an")
        command.append(str(self.temporary))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.width = width
        self.height = height

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"renderer frame shape changed: {frame.shape} != {(self.height, self.width, 3)}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while receiving video frames") from exc

    def finish(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        error = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg video encoder failed ({return_code}): {error.strip()}")

    def commit(self) -> None:
        os.replace(self.temporary, self.destination)

    def abort(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
        finally:
            for stream in (self.process.stdin, self.process.stderr):
                if stream is not None:
                    with suppress(BaseException):
                        stream.close()
            if self.temporary.exists():
                self.temporary.unlink()


def render_artifacts(
    *,
    source: str | Path,
    targets: list[RenderTarget],
    settings: dict[str, Any],
    analysis_result: dict[str, Any],
    progress: Callable[[int, int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    _raise_if_cancelled(is_cancelled)
    source_path = Path(source).expanduser().resolve()
    if not targets:
        raise ValueError("at least one render target is required")
    resolved_targets = [RenderTarget(target.mode, target.path.resolve()) for target in targets]
    if not paths_are_distinct([source_path, *(target.path for target in resolved_targets)]):
        raise ValueError("input and every render output path must be distinct")
    for target in resolved_targets:
        if target.mode not in {"debug", "redacted"}:
            raise ValueError(f"unsupported render mode: {target.mode}")

    expected_source = analysis_result["source_video"]
    metadata = expected_source["metadata"]
    source_metadata = probe_video(source_path)
    expected_width = int(metadata["width"])
    expected_height = int(metadata["height"])
    if (source_metadata.width, source_metadata.height) != (
        expected_width,
        expected_height,
    ):
        raise ValueError(
            "source video dimensions do not match the PrivateFrame result: "
            f"{source_metadata.width}x{source_metadata.height} != "
            f"{expected_width}x{expected_height}"
        )
    expected_fps = float(metadata["fps"])
    if not math.isclose(
        source_metadata.fps,
        expected_fps,
        rel_tol=0.01,
        abs_tol=0.01,
    ):
        raise ValueError(
            "source video FPS does not match the PrivateFrame result: "
            f"{source_metadata.fps:g} != {expected_fps:g}"
        )
    _raise_if_cancelled(is_cancelled)
    observations = analysis_result["observations"]
    recognition = analysis_result.get("recognition")
    recognition_policy = settings.get("recognition_policy", {"mode": "all", "unknown_action": "blur"})
    by_frame: dict[int, list[tuple[dict[str, Any], bool, str]]] = {}
    blurred_observations = 0
    kept_observations = 0
    fail_safe_observations = 0
    for item in observations:
        should_blur, reason = _identity_should_blur(item, recognition_policy, recognition)
        by_frame.setdefault(int(item["frame_idx"]), []).append((item, should_blur, reason))
        if should_blur:
            blurred_observations += 1
            if reason in {"unknown_blur", "conflict_blur"}:
                fail_safe_observations += 1
        else:
            kept_observations += 1

    width, height = expected_width, expected_height
    fps = expected_fps
    expected_frames = int(metadata["frame_count"])
    if progress is not None:
        progress(0, expected_frames, "render")
    _raise_if_cancelled(is_cancelled)
    audio = settings.get("audio", {})
    writer_type = _PyAVWriter if str(settings.get("backend", "pyav")) == "pyav" else _FFmpegWriter
    pyav_writer = writer_type is _PyAVWriter
    writers: list[_VideoWriter] = []
    try:
        for target in resolved_targets:
            writers.append(
                writer_type(
                    target.path,
                    source_path,
                    width,
                    height,
                    fps,
                    settings,
                    str(audio.get(target.mode, "none")),
                )
            )
    except BaseException:
        _abort_writers(writers, pyav=pyav_writer)
        raise
    count = 0
    try:
        for frame_idx, _timestamp, _pts, frame in iter_oriented_frames(source_path):
            _raise_if_cancelled(is_cancelled)
            for target_index, target in enumerate(resolved_targets):
                output = frame.copy()
                for item, should_blur, _reason in by_frame.get(frame_idx, []):
                    if target.mode == "debug":
                        _debug_box(output, item, settings)
                    elif should_blur:
                        _blur(output, item["box"], settings)
                if target.mode == "debug":
                    _debug_frame_number(output, frame_idx)
                writers[target_index].write(output)
            count += 1
            if progress is not None:
                progress(count, expected_frames, "render")
            _raise_if_cancelled(is_cancelled)
        if count != expected_frames:
            raise RuntimeError(f"rendered {count} frames, analysis manifest declares {expected_frames}")
        for index in range(len(writers)):
            writers[index].finish()
        for index in range(len(writers)):
            writers[index].commit()
    except BaseException:
        _abort_writers(writers, pyav=pyav_writer)
        raise

    outputs = []
    for target in resolved_targets:
        output_metadata = probe_video(target.path)
        outputs.append(
            {
                "mode": target.mode,
                "path": str(target.path),
                "sha256": sha256_file(target.path),
                "metadata": output_metadata.to_dict(),
            }
        )
    result = {
        "frame_count": count,
        "observations": len(observations),
        "blurred_observations": blurred_observations,
        "kept_observations": kept_observations,
        "fail_safe_observations": fail_safe_observations,
        "seconds": time.perf_counter() - started,
        "outputs": outputs,
    }
    if pyav_writer:
        # PyAV stream/container wrappers can participate in native reference
        # cycles even after every container has been closed.  Reclaim them
        # while Python and FFmpeg are still fully initialized; otherwise some
        # macOS PyAV builds can defer destruction to interpreter shutdown and
        # race a native recursive-mutex teardown after a successful render.
        _release_native_writer_cycles(writers, pyav=True)
    return result


__all__ = ["RenderTarget", "_identity_should_blur", "render_artifacts"]
