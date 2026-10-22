"""Curated configuration guidance for everyday CLI discovery.

Defaults and accepted paths come from Base; these descriptions explain choices
without making the discovery subset an allowlist for configuration parsing.
"""

from __future__ import annotations

from typing import Any


def _option(description: str, when_to_use: str, tradeoff: str, **constraints: Any) -> dict[str, Any]:
    return {
        "description": description,
        "when_to_use": when_to_use,
        "tradeoff": tradeoff,
        **constraints,
    }


DESCRIBE_OPTION_GROUPS = [
    {
        "id": "models",
        "description": "Model package, local model storage, and inference hardware.",
        "fields": ["models.name", "models.root", "runtime.provider"],
    },
    {
        "id": "analysis",
        "description": "Sampling speed and treatment of frames between detections.",
        "fields": ["scan.max_analysis_fps", "tracking.between_scan_frames"],
    },
    {
        "id": "privacy",
        "description": "Which people to blur using reference photos, and how to handle uncertain matches.",
        "fields": [
            "recognition.mode", "recognition.reference_dir", "recognition.unknown_action",
            "recognition.profile", "recognition.similarity_threshold",
        ],
    },
    {
        "id": "redaction",
        "description": "Face-region coverage, blur strength, and mosaic size.",
        "fields": [
            "render.redaction.method", "render.redaction.box_scale",
            "render.redaction.gaussian.algorithm", "render.redaction.gaussian.kernel_ratio",
            "render.redaction.gaussian.min_kernel", "render.redaction.mosaic.block_size_ratio",
            "render.redaction.mosaic.min_block_size",
        ],
    },
    {
        "id": "video",
        "description": "Video compatibility, encoding speed, quality, and file size.",
        "fields": [
            "render.video_output.backend", "render.video_output.encoder",
            "render.video_output.pixel_format", "render.video_output.preset",
            "render.video_output.rate_control.mode", "render.video_output.rate_control.quality",
            "render.video_output.rate_control.bitrate", "render.video_output.rate_control.max_bitrate",
            "render.video_output.rate_control.buffer_size", "render.video_output.keyframe_interval",
            "render.video_output.faststart",
        ],
    },
    {
        "id": "audio",
        "description": "Whether to retain audio and how to encode it.",
        "fields": ["render.video_output.audio.redacted", "render.video_output.audio.bitrate"],
    },
]


DESCRIBE_OPTION_METADATA = {
    "models.name": _option(
        "Manifest-backed face model package; raccoon_s is the default.",
        "Use raccoon_s for routine processing; evaluate raccoon_l when detection coverage matters more.",
        "The larger package needs more storage and compute; compare representative output before switching.",
    ),
    "models.root": _option(
        "InsightFace root containing models/<package>; this root is authoritative and no alternate root is searched.",
        "Change when models are stored on another disk or shared with an existing local installation.",
        "A root without the selected package can trigger a download during execution; dry-run only checks it.",
    ),
    "runtime.provider": _option(
        "Inference execution provider; auto chooses CoreML, then CUDA, then CPU when available.",
        "Keep auto normally; select an installed provider to compare hardware or diagnose a backend issue.",
        "Availability depends on the installed ONNX Runtime; forcing an unavailable provider fails validation.",
    ),
    "scan.max_analysis_fps": {
        # The full sampling explanation and tuning guidance live in the shared metadata.
        "when_to_use": "Keep the default Fast mode (15) for faster processing; use 30 for greater temporal coverage, fast motion, or frequent occlusion.",
        "tradeoff": "Wider sampling gaps can miss briefly visible faces. Higher sampling costs more compute and is not a detection guarantee.",
    },
    "tracking.between_scan_frames": _option(
        "How to locate existing face tracks between sampled detection frames: interpolate or visual tracking/review.",
        "Keep interpolate for speed; try visual when motion makes interpolated boxes inaccurate. It matters only when sampling skips frames.",
        "Visual adds processing work. Neither option guarantees discovery of a new face visible only between full-frame scans.",
    ),
    "recognition.mode": _option(
        "all blurs every detected face. blur_only blurs people matched to the reference photos; exempt keeps those people visible. With unknown_action=auto, unmatched or uncertain people stay visible in blur_only and are blurred in exempt.",
        "Keep all for everyone; use blur_only for only the people in reference photos, or exempt to keep those people visible. Both photo modes require reference_dir.",
        "Photo matching adds processing work. In blur_only with the default auto policy, a target person who is not recognized can remain visible.",
    ),
    "recognition.reference_dir": _option(
        "Local folder of JPG, JPEG, PNG, or WebP reference photos, without person subfolders or names. Only the largest detected face in each photo is considered; unusable photos are skipped with a reason.",
        "Provide a non-symlink folder for blur_only or exempt. Different people and multiple photos of the same person can share this folder. Mode all does not read it.",
        "Use clear single-person photos: other faces in a group photo are ignored. Both photo modes fail if no photo provides a usable face.",
        format="local_directory_path",
    ),
    "recognition.unknown_action": _option(
        "Treatment of unmatched or uncertain people: auto keeps them visible in blur_only and blurs them in exempt; blur or keep explicitly selects the action. Mode all always blurs every detected face.",
        "Keep auto to follow the chosen photo mode. Set blur when uncertain people must also be obscured, or keep when they should remain visible.",
        "keep can leave an unrecognized target visible. Model loading and inference errors stop the task instead of becoming ordinary unmatched faces. The effective policy is saved in analysis JSON for later rendering.",
    ),
    "recognition.profile": _option(
        "Default recognition sampling caps: fast, balanced, or accurate selects at most 1, 3, or 5 eligible samples per canonical track unless the advanced sample-count override is set.",
        "Keep balanced for selective modes; choose fast to reduce recognition work or accurate to collect more evidence.",
        "More samples cost more compute and do not guarantee confirmation. Mode all does not load the recognizer.",
    ),
    "recognition.similarity_threshold": _option(
        "Base cosine-similarity gate for matching a person to the reference photos; additional temporal and consensus rules also apply.",
        "Keep 0.40 normally; adjust only after checking reference photos and representative video results.",
        "Lower values can match the wrong person. Higher values can miss a target; unmatched faces follow unknown_action.",
        unit="cosine_similarity", finite=True,
    ),
    "render.redaction.method": _option(
        "Redact detected face regions with gaussian blur or mosaic pixels.",
        "Choose the requested visual style; existing analysis JSON can be rendered again without rerunning models.",
        "Appearance changes, but detection coverage does not; review the chosen strength on representative faces.",
    ),
    "render.redaction.box_scale": _option(
        "Scale each final face box around its center; 1.0 uses the detected region and values above 1 add coverage.",
        "Increase when the mask needs more margin around face edges; keep 1.0 for the standard coverage.",
        "Larger values obscure more surrounding content; values below 1 can leave parts of a face visible.",
        unit="box_size_multiplier",
    ),
    "render.redaction.gaussian.algorithm": _option(
        "pyramid applies Gaussian blur on a smaller working image; exact applies it at full region resolution.",
        "Keep pyramid for faster rendering; use exact when comparing blur appearance or investigating a rendering difference.",
        "Exact can be substantially slower for large faces; both use the configured relative blur strength.",
    ),
    "render.redaction.gaussian.kernel_ratio": _option(
        "Gaussian kernel size relative to the larger face-region dimension, subject to min_kernel and odd-size rounding.",
        "Adjust Gaussian blur strength; increase for stronger blur and inspect results before reducing it.",
        "The minimum kernel can mask changes on small faces. Stronger blur may cost more rendering time.",
        unit="face_size_ratio", exclusive_minimum=0.0, finite=True,
    ),
    "render.redaction.gaussian.min_kernel": _option(
        "Minimum Gaussian kernel size in face-region pixels; must be a positive odd integer.",
        "Adjust the blur-strength floor for small faces when kernel_ratio alone has little effect.",
        "A lower floor can retain more facial detail; a larger floor can increase blur and rendering work.",
        unit="pixels", minimum=1, constraints=["Must be an odd integer."],
    ),
    "render.redaction.mosaic.block_size_ratio": _option(
        "Mosaic block size relative to the larger face-region dimension, subject to min_block_size.",
        "Increase for coarser pixelation; decrease only when a finer mosaic is acceptable.",
        "Smaller blocks retain more facial detail; the minimum block size can mask changes on small faces.",
        unit="face_size_ratio", exclusive_minimum=0.0, maximum=1.0, finite=True,
    ),
    "render.redaction.mosaic.min_block_size": _option(
        "Minimum mosaic block size in face-region pixels.",
        "Adjust pixelation strength on small faces when block_size_ratio alone has little effect.",
        "Smaller blocks reveal more detail; larger blocks produce a coarser mask.",
        unit="pixels", minimum=1,
    ),
    "render.video_output.backend": _option(
        "Video writer backend: in-process PyAV or an installed ffmpeg executable.",
        "Keep pyav normally; choose ffmpeg for audio transcoding or encoder workflows unavailable through PyAV.",
        "FFmpeg requires a usable executable and codecs. Validate the complete encoder/audio combination with dry-run.",
    ),
    "render.video_output.encoder": _option(
        "Video encoder name, such as libx264; it must be available to the selected backend.",
        "Change for a required codec or a hardware encoder; inspect availability using doctor or dry-run.",
        "A different encoder may require different presets, pixel formats, and rate-control settings.",
        min_length=1, dynamic_values="Encoders available to the selected backend; validate with command --dry-run.",
    ),
    "render.video_output.pixel_format": _option(
        "Encoded pixel format; yuv420p is the default for broad playback compatibility.",
        "Change only when an encoder or target playback workflow requires another format.",
        "Some formats or frame dimensions are incompatible with the encoder or player; verify with dry-run.",
        min_length=1, dynamic_values="Pixel formats supported by the selected encoder and video dimensions.",
    ),
    "render.video_output.preset": _option(
        "Encoder-specific speed/compression preset; an empty string omits an explicit preset.",
        "For libx264, try faster presets when rendering speed matters or slower presets for better compression.",
        "Preset names vary by encoder. Faster encoding can require larger files for comparable quality.",
        dynamic_values="Preset names supported by the selected encoder; validate with command --dry-run.",
    ),
    "render.video_output.rate_control.mode": _option(
        "Quality-based crf/cq or bitrate-based vbr/cbr control. Support depends on the encoder.",
        "Keep crf with libx264 for routine files; use vbr/cbr when a delivery target requires bitrate control.",
        "Replace the whole rate_control mapping when switching modes. Individual dotted leaves merge with inherited fields and can leave an invalid quality/bitrate combination.",
        constraints=["crf/cq require exactly mode and quality.", "vbr requires mode and bitrate, with optional max_bitrate.", "cbr requires mode and bitrate, with optional buffer_size."],
        examples=[
            {"argv": ["--render.video_output.rate_control", "{mode: vbr, bitrate: 4M}"]},
            {"argv": ["--render.video_output.rate_control", "{mode: crf, quality: 18}"]},
        ],
    ),
    "render.video_output.rate_control.quality": _option(
        "Quality level in crf/cq mode; the application accepts integers from 0 through 51.",
        "With the default libx264 encoder, keep 23 for balanced quality and file size; choose 18 to retain more detail or 28 for smaller files. Lower values increase quality and file size.",
        "This changes video encoding, not face detection. Quality scale behavior and availability depend on the encoder; omit it in vbr/cbr mode.",
        unit="encoder_quality_level",
    ),
    "render.video_output.rate_control.bitrate": _option(
        "Target video bitrate for vbr/cbr, as a positive integer or a string with k/m/g suffix.",
        "Set together with vbr/cbr mode when targeting a delivery bitrate or approximate file size.",
        "Too little bitrate loses detail. Quality-based modes reject bitrate fields; the selected encoder must support the rate-control combination.",
        unit="bits_per_second", exclusive_minimum=0,
    ),
    "render.video_output.rate_control.max_bitrate": _option(
        "Optional peak video bitrate for vbr mode, using the same units and syntax as bitrate.",
        "Set when the VBR delivery target has a peak-rate constraint.",
        "A tight peak limit can reduce quality during complex scenes; this field is not accepted in crf/cq/cbr mode.",
        unit="bits_per_second", exclusive_minimum=0,
    ),
    "render.video_output.rate_control.buffer_size": _option(
        "Optional rate-control buffer size for cbr mode, as a positive integer or k/m/g string.",
        "Set when the CBR delivery specification requires a particular buffer size.",
        "Buffer behavior is encoder-dependent; this field is not accepted in crf/cq/vbr mode.",
        unit="bits", exclusive_minimum=0,
    ),
    "render.video_output.keyframe_interval": _option(
        "Requested video keyframe interval in frames; 0 omits an explicit GOP interval.",
        "Adjust for seeking, editing, or a delivery specification; keep the default unless one of these needs applies.",
        "Shorter intervals can increase file size; actual keyframe placement also depends on the encoder.",
        unit="frames",
    ),
    "render.video_output.faststart": _option(
        "Request MP4 metadata near the beginning of the file for progressive playback.",
        "Keep enabled for videos that will be viewed before a download completes.",
        "Finalization can require extra I/O. PyAV's audio remux always enables faststart, even when this flag is false; frame rate and face detection are unaffected.",
    ),
    "render.video_output.audio.redacted": _option(
        "none omits audio, copy remuxes compatible source audio, and aac requests AAC output.",
        "Choose none for silent output. PyAV's aac mode only remuxes already-AAC source audio; use ffmpeg to transcode other source codecs to AAC.",
        "Retained audio may contain names or other identifying content. Copy requires MP4-compatible audio; a source without audio remains silent.",
    ),
    "render.video_output.audio.bitrate": _option(
        "Requested AAC audio bitrate when the ffmpeg backend transcodes audio; accepts a positive integer or k/m/g string.",
        "Adjust audio quality/file size when using ffmpeg with audio.redacted=aac.",
        "It does not change copied audio or PyAV's AAC remux; lower transcoding bitrates can reduce audio quality.",
        unit="bits_per_second", exclusive_minimum=0,
    ),
}
