"""Generate the bundled, complete PrivateFrame configuration reference offline.

Run ``python -m insightface.app.privateframe.config_reference --check`` to check
that the checked-in documentation matches current defaults and public fields.
No model resolution, media probing, or inference is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping

DEFAULT_REFERENCE_PATH = Path(__file__).with_name("docs") / "configuration.md"
_DESCRIPTIONS: dict[str, str] = {}


def _register(prefix: str, lines: str) -> None:
    for line in lines.strip().splitlines():
        key, description = line.strip().split("|", 1)
        _DESCRIPTIONS[f"{prefix}.{key}" if prefix else key] = description.strip()


_register(
    "",
    """
output.artifacts_level|Retain only the reusable result in final mode; audit also writes development evidence and summaries, while debug retains additional intermediate records.
models.name|Select the manifest-backed face detector, verifier, and optional recognizer package.
models.root|Local InsightFace root containing models/<package>; this selected root is authoritative.
models.detection.nms_iou_threshold|IoU threshold used to suppress overlapping detections within the detector.
models.detection.max_detections|Maximum detections retained by each detector invocation.
runtime.provider|Inference execution provider; auto follows the available CoreML, CUDA, then CPU preference.
runtime.scrfd_static_shape_sessions|Reuse a separate fixed-input-shape SCRFD session for each detector resolution; false selects the dynamic-session compatibility path.
runtime.intra_op_threads|ONNX Runtime thread count within an operator.
runtime.inter_op_threads|ONNX Runtime thread count across operators.
recognition.mode|all blurs every detected face without recognition; blur_only blurs people matched to reference photos; exempt keeps matched people visible. With unknown_action=auto, unmatched or uncertain people remain visible in blur_only and are blurred in exempt.
recognition.reference_dir|Local folder of JPG, JPEG, PNG, or WebP reference photos, with no person subfolders or names. Each photo contributes only its largest detected face; an unusable largest face causes the photo to be skipped. Multiple people and multiple photos per person may share one folder. Both selective modes require at least one usable reference face. all does not read this folder.
recognition.unknown_action|Action for unmatched or uncertain people: auto resolves to keep in blur_only and blur in exempt; blur or keep explicitly overrides that action. all always blurs every detected face. keep can leave an unrecognized target visible. The effective policy is stored in analysis JSON and reused during rendering; inference errors fail the task.
recognition.profile|Set the default cap of 1, 3, or 5 eligible temporal identity samples per track for fast, balanced, or accurate; max_frames_per_track can override this cap.
recognition.similarity_threshold|Base cosine similarity gate for identity confirmation; temporal-evidence offsets and other recognition gates still apply.
recognition.max_frames_per_track|Optional explicit limit of 1–32 identity samples per track; absent or null uses the recognition profile.
scan.max_analysis_fps|Default Fast mode targets 15 regular detector samples per second of input video; this is a soft ceiling, integer stride uses 5% tolerance, and forced scans may exceed it. Use 30 for greater temporal coverage. This is not a wall-clock throughput target, and output FPS is unchanged.
scan.session_sharing|Permit concurrent or serialized calls to the shared full-frame detector session.
scan.workers|Number of concurrent full-frame scanning workers.
scan.pipeline_depth|Number of video frames allowed in the scan pipeline queue.
scan.passes|Ordered array of full-frame detector views; each view declares its own angles, input size, padding, and confidence gate.
scan.global_nms_iou|IoU threshold for suppressing duplicate boxes after combining scan views.
scan.containment_threshold|Containment threshold used when merging overlapping scan-view detections.
streaming.max_missed_seconds|Maximum elapsed video time for retaining a track without a detector match.
streaming.max_retroactive_seconds|Historical video-time horizon available for retrospective track replay.
streaming.recent_frame_cache_frames|Decoded-frame retention target; null derives it from replay horizons, zero disables retention, and an explicit count limits retained frames.
streaming.recent_frame_cache_max_bytes|Hard byte limit for retained decoded BGR frames, independent of the frame target.
streaming.pre_roll_decode_chunk_frames|Maximum number of pre-roll frames decoded together while replaying history in reverse blocks.
streaming.corridor_expansion|Face-relative expansion of the crop corridor used for historical tracking.
streaming.pre_roll_corridor_expansion|Face-relative expansion of the crop corridor used for leading pre-roll replay.
streaming.max_corridor_side_pixels|Maximum source-image side length of a historical crop corridor, in pixels.
streaming.eviction_interval_frames|Interval in video frames between encoded-packet cache eviction checks.
streaming.progress_every_frames|Interval in processed video frames for streaming progress diagnostics.
tracking.between_scan_frames|interpolate publishes geometry between detector anchors; visual runs optical flow and local review on skipped scans. Stride 1 has no sampled interval and uses the visual path.
tracking.association_max_scan_gap|Maximum gap measured in actual detector opportunities, including forced scans; sampled-out frames do not consume this budget.
tracking.association_max_gap_seconds|Independent elapsed video-time ceiling for detector association.
tracking.association_strict_geometry_after_seconds|Anchor gap after which association requires the stricter overlap/center corridor; must not exceed association_max_gap_seconds.
tracking.association_min_iou|Minimum IoU gate for associating a detector box with an existing track.
tracking.association_max_center_distance|Base normalized center-distance allowance for detector-to-track association.
tracking.association_sparse_max_center_distance|Upper normalized center-distance allowance applied to sparse detector association.
tracking.association_max_area_ratio|Maximum area ratio allowed between associated track and detector boxes.
tracking.association_min_score|Minimum combined geometry score accepted by detector-to-track association.
tracking.endpoint_extension|Initial leading/trailing endpoint replay allowance for tracks, in frames.
tracking.reliable_endpoint_extension|Trailing endpoint replay allowance after reliable detector support, in frames.
tracking.reliable_pre_roll_extension|Leading pre-roll replay allowance for reliable tracks, in frames.
tracking.reliable_endpoint_min_detector_frames|Number of detector-supported frames required before the longer reliable endpoint allowances apply.
tracking.endpoint_conflicts.enabled|Resolve redundant unanchored endpoints from same-frame face-location evidence, only in recognition.mode=all. Set false for an A/B baseline. First-stage resolution does not stop tracking early; selective photo modes retain their own track decisions.
tracking.endpoint_conflicts.angles|Nonempty list of distinct rotations for extra duplicate-box review: 0, 90, or -90 degrees. Default [0] runs one upright detection per requested crop; a frame may require multiple crops. Use [0,90,-90] to review rotated faces more thoroughly at higher cost. Changing angles can retain tilted-face duplicates or change which overlapping boxes are removed and their coverage at face edges; review the output when switching. Ordinary tracking review and full-frame scan angles are independent; recheck_min_frame_gap controls the frame interval separately.
tracking.endpoint_conflicts.angles.*|Rotation in degrees for this extra duplicate-box review view: 0, 90, or -90, without duplicates. Default [0] uses one upright detection per requested crop; [0,90,-90] costs three calls per crop and can resolve more tilted-face duplicates. Changing angles can also change duplicate removal and coverage at face edges; review the output when switching. Ordinary tracking review and full-frame scan angles are unchanged by this setting.
tracking.endpoint_conflicts.nearby_iou|Minimum box IoU proposing a nearby endpoint conflict, in [0,1]. This is a cheap screening gate, not proof that the two boxes represent the same face.
tracking.endpoint_conflicts.nearby_center_distance|Alternative nearby-conflict center-distance gate, normalized by the square root of the smaller box area; must be positive and finite.
tracking.endpoint_conflicts.match_min_iou|Minimum IoU for associating an independently measured local face with a shared same-frame detection, in [0,1]. Predicted-box to target-local matching uses revalidation.match_* geometry, including IoU OR containment. An unmatched candidate is not automatically a duplicate.
tracking.endpoint_conflicts.match_max_center_distance|Maximum local-face/shared-detection center distance, normalized by the square root of the smaller box area; must be positive and finite. Predicted-box to target-local matching instead uses revalidation.match_max_center_distance.
tracking.endpoint_conflicts.match_max_area_ratio|Maximum larger/smaller box-area ratio for local-face/shared-detection matching and retained-owner coverage; must be finite and at least 1. Predicted-box to target-local matching instead uses revalidation.match_max_area_ratio.
tracking.endpoint_conflicts.match_min_confidence|Minimum detector confidence for positively associating face-location evidence in both independent target-local selection and shared-instance matching, in [0,1]. This is not a cutoff for ignoring other returned faces: weaker second-face evidence can still block removal when no accepted core box covers it. Verifier probability alone does not establish a face location.
tracking.endpoint_conflicts.match_min_margin|Minimum best-versus-runner-up geometric match-score margin for both target-local selection and shared-instance matching, in [0,1]. Ambiguous candidates remain unresolved instead of being removed.
tracking.endpoint_conflicts.equivalence_iou|Strong IoU gate in (0,1] for identifying two independently detected locations as equivalent evidence. It does not compare final blur boxes or infer that unmatched or occluded faces are duplicates.
tracking.endpoint_conflicts.recheck_min_frame_gap|Minimum positive number of source frames between extra reviews for an unresolved conflict. A skipped review never authorizes applying another frame's duplicate decision; complete same-frame cached evidence can still be reused.
tracking.endpoint_conflicts.max_calls_per_frame|Nonnegative maximum additional SCRFD angle/view calls on one source frame. Zero allows evidence reuse without extra inference. A multi-angle review requires multiple calls; an incomplete review cannot establish a duplicate.
tracking.endpoint_conflicts.max_calls_total|Nonnegative absolute ceiling on extra SCRFD angle/view calls for the video. The effective video budget is min(max_calls_total, ceil(source duration in seconds * max_calls_per_video_second)); zero disables extra inference.
tracking.endpoint_conflicts.max_calls_per_video_second|Nonnegative finite extra SCRFD angle/view call allowance per second of source-video duration, capped by max_calls_total. This allocates work over the whole video, not by wall-clock speed or a sliding time window. Zero disables extra inference.
tracking.endpoint_conflicts.cache_entries|Nonnegative maximum cached local evidence entries, including empty results. Keys include the source frame and exact ROI/model/view settings; geometry and landmarks are cached without copying video pixels. Zero disables this cache. Entries cannot substitute for another frame's evidence.
tracking.long_gap_requires_continuous_flow|Require continuous optical-flow evidence when associating across a long detector gap.
tracking.long_gap_min_iou|Stricter endpoint IoU requirement used for long-gap association.
tracking.long_gap_max_center_distance|Stricter normalized center-distance ceiling used for long-gap association.
""",
)
_register(
    "scan.passes.*",
    """
name|Unique name identifying this detector view in diagnostics.
angles|Nonempty array of detector-view rotations in degrees; supported values are 0, 90, and -90.
angles.*|Rotation in degrees for this indexed detector-view entry; supported values are 0, 90, and -90.
input_size|Square detector input side in pixels; a positive multiple of 32.
horizontal_padding_ratio|Horizontal image padding as a fraction of source width; nonnegative and at most 1.
vertical_padding_ratio|Vertical image padding as a fraction of source height; nonnegative and at most 1.
confidence_threshold|Minimum full-frame detector confidence for this view, within [0, 1].
candidate_filter.enabled|Enable raw-box geometry filtering for this scan view before clipping and local review.
candidate_filter.min_box_area_fraction|Minimum raw candidate box area divided by source frame area, within [0, 1], when this filter is enabled.
candidate_filter.min_height_width_ratio|Minimum raw candidate height/width ratio when this filter is enabled; must be positive.
candidate_filter.max_height_width_ratio|Optional maximum raw height/width ratio; absent or null omits the maximum, otherwise it must be at least the minimum ratio.
candidate_filter.aspect_ratio_exempt_min_area_fraction|Raw box/frame area fraction above which the aspect-ratio gate is bypassed; must be between min_box_area_fraction and 1.
""",
)
_register(
    "scan.scene_cut_detector",
    """
signature_size|Side length of the downscaled grayscale scene signature, in pixels; at least 32.
history_frames|Number of preceding frame differences retained for the adaptive scene-change baseline; positive.
min_mean_absdiff|Steady-state floor for mean absolute grayscale difference, in signature intensity units; nonnegative.
bootstrap_min_mean_absdiff|Scene-change threshold before a history baseline is available; at least min_mean_absdiff.
relative_multiplier|Multiplier on the recent scene-difference baseline; must exceed 1.
relative_offset|Nonnegative additive offset to the adaptive scene-difference threshold.
max_corners|Maximum corner count for signature-scale optical-flow continuity checks; at least min_corners.
min_corners|Minimum corner count needed for a flow-continuity estimate; at least 4.
quality_level|Relative corner-quality threshold for goodFeaturesToTrack; within (0, 1].
min_distance|Minimum corner separation on the scene signature, in pixels; positive.
block_size|Corner-detector neighborhood side in signature pixels; odd and at least 3.
lk_window_size|Lucas–Kanade window side in signature pixels; odd and at least 3.
lk_max_level|Highest Lucas–Kanade pyramid level; nonnegative.
lk_max_iterations|Maximum iterations of scene-continuity Lucas–Kanade flow; positive.
lk_epsilon|Positive motion convergence threshold for scene-continuity Lucas–Kanade iterations.
lk_min_eigenvalue|Positive minimum eigenvalue threshold for scene-continuity Lucas–Kanade points.
max_forward_backward_error|Maximum accepted return error of forward/backward signature flow, in pixels; positive.
ransac_reprojection_threshold|RANSAC residual threshold for signature-flow affine continuity, in pixels; positive.
max_flow_inlier_fraction|Maximum affine-consistent corner fraction compatible with a primary scene-cut confirmation, within [0, 1].
appearance_confirmation.enabled|Enable the strong appearance-discontinuity conjunction for cuts with a few accidental flow inliers.
appearance_confirmation.min_mean_absdiff|Minimum nonnegative signature intensity difference for the strong appearance-confirmation path.
appearance_confirmation.max_histogram_correlation|Upper grayscale histogram correlation for appearance confirmation, within [-1, 1].
appearance_confirmation.max_spatial_correlation|Upper spatial signature correlation for appearance confirmation, within [-1, 1].
appearance_confirmation.max_flow_inlier_fraction|Upper affine-consistent corner fraction for appearance confirmation, within [0, 1].
flash_suppression.enabled|Suppress A–B–A exposure flashes using one-frame lookahead before declaring a scene boundary.
flash_suppression.max_skip_mean_absdiff|Maximum nonnegative difference between the two outer signatures of an A–B–A flash candidate.
flash_suppression.max_skip_to_transition_ratio|Maximum outer-frame difference relative to the flash transition; within (0, 1].
flash_suppression.min_skip_spatial_correlation|Minimum spatial similarity of the two outer flash-candidate signatures, within [-1, 1].
""",
)
_register(
    "tracking.fragment_stitching",
    """
enabled|Enable reconciliation of track fragments supported by compatible local geometry.
resolve_duplicate_candidates_before_stabilization|Resolve competing duplicate candidates before output box stabilization.
min_local_confidence|Minimum local detector confidence contributing to a fragment agreement.
min_local_iou|Minimum local-box IoU for a fragment agreement.
min_overlap_frames|Minimum number of overlapping video frames examined for fragment agreement.
min_agreement_frames|Minimum number of agreeing overlap frames required for stitching.
min_agreement_fraction|Minimum fraction of overlap evidence that must agree before fragments are stitched.
max_interval_gap_frames|Maximum gap in video frames between candidate fragment intervals.
""",
)
_register(
    "tracking.kalman_optical_flow",
    """
roi_size|Working square side length for local tracking images, in pixels.
roi_expansion|Face-relative expansion used to build the optical-flow tracking ROI.
roi_edge_padding_pixels|Working-ROI edge margin in pixels inside which estimated flow scale is held at 1 to avoid edge-induced size changes.
max_source_canvas_side_pixels|Maximum side length of the source crop materialized for a tracking ROI, in pixels.
max_points|Maximum feature-point count retained for local optical flow.
min_points|Minimum valid feature-point count required for ordinary optical-flow motion.
pyramid_levels|Number of scales used by the local optical-flow pyramid.
window_radius|Radius of the local optical-flow comparison window in working-image pixels.
max_iterations|Maximum iterations used by the local optical-flow solver.
termination_epsilon|Motion convergence threshold used by the local optical-flow solver.
forward_backward_max_error|Maximum feature return error for forward/backward flow consistency, in working-image pixels.
residual_threshold|Maximum feature-motion residual for accepting local flow/affine inliers, in working-image pixels.
feature_box_expansion|Fractional expansion of the face box used for selecting flow features.
max_scale_change|Maximum fractional scale change permitted by the local motion estimate.
max_coast_frames|Maximum consecutive frames to coast without a trustworthy flow measurement.
require_cycle_consistency_after_coast|Require a forward/backward recovery cycle before accepting geometry after coasting.
recovery_cycle_min_iou|Minimum IoU between recovered and cycle-returned boxes after a coast.
endpoint_affine_repair.enabled|Try a strict partial-affine fallback after ordinary visual-mode endpoint flow stops.
endpoint_affine_repair.max_frames|Maximum additional visual-mode affine fallback frames; interpolate uses the shared endpoint limits instead.
process_noise_position|Kalman process-noise weight for position state.
process_noise_velocity|Kalman process-noise weight for velocity state.
detector_measurement_noise|Kalman observation-noise weight for detector geometry.
flow_measurement_noise|Kalman observation-noise weight for optical-flow geometry.
flow_updates_size|Allow optical-flow measurements to update width/height; false restricts flow updates to center translation.
""",
)
_register(
    "tracking.kalman_optical_flow.bidirectional_fusion",
    """
max_gap_frames|Maximum detector-bounded gap replayed in both directions, in frames; positive.
max_materialized_bytes|Maximum bytes of decoded corridor data materialized for bidirectional replay; at least 1 MiB.
corridor_expansion|Expansion of the shared bidirectional replay corridor; must exceed 1.
max_corridor_side_pixels|Maximum side of the shared source corridor, in pixels; must cover the tracking roi_size.
geometry_bridge.enabled|Enable the guarded geometry bridge for compatible forward and reverse paths.
geometry_bridge.edge_epsilon_pixels|Nonnegative image-edge tolerance in pixels when detecting truncated endpoint geometry.
geometry_bridge.min_edge_expansion_ratio|Minimum expansion ratio indicating an edge-truncated box; must exceed 1.
geometry_bridge.min_both_trusted_fraction|Minimum replay fraction with trustworthy estimates from both directions, within [0, 1].
geometry_bridge.min_mutual_consistent_fraction|Minimum replay fraction with mutually consistent directional geometry, within [0, 1].
association_rescue.enabled|Permit a strict bidirectional endpoint check to rescue a failed ordinary association.
association_rescue.max_endpoint_center_speed|Maximum normalized endpoint center displacement divided by the endpoint gap in video frames for an association rescue; positive.
association_rescue.max_area_ratio|Maximum endpoint area ratio for an association rescue; at least 1.
association_rescue.min_endpoint_iou|Minimum endpoint overlap for an association rescue, within [0, 1].
association_rescue.max_endpoint_center_distance|Maximum normalized endpoint center distance for an association rescue; positive.
mutual_min_iou|Minimum IoU for forward/reverse geometry agreement, within [0, 1].
mutual_max_center_distance|Maximum normalized center separation for forward/reverse agreement; positive.
anchor_max_center_distance|Maximum normalized separation from the detector anchor when validating a directional path; positive.
anchor_max_area_ratio|Maximum area ratio relative to a detector anchor; at least 1.
local_review_min_confidence|Minimum local detector confidence used to validate bidirectional geometry, within [0, 1].
local_review_min_iou|Minimum local-review IoU required by bidirectional validation, within [0, 1].
local_review_min_iou_margin|Minimum IoU advantage of the preferred local-review candidate, within [0, 1].
soft_bias_beta|Nonnegative strength of the smooth confidence-dependent directional weighting.
soft_bias_radius|Nonnegative temporal neighborhood radius used to smooth directional bias.
soft_confidence_low|Lower local-confidence endpoint for smooth directional weighting; below soft_confidence_high within [0, 1].
soft_confidence_high|Upper local-confidence endpoint for smooth directional weighting; above soft_confidence_low within [0, 1].
""",
)
_register(
    "revalidation",
    """
input_size|Square local-review detector input side in pixels; a positive multiple of 32.
angles|Nonempty local-review rotation array in degrees; supported values are 0, 90, and -90.
angles.*|Rotation in degrees for this indexed local-review view; supported values are 0, 90, and -90.
confidence_threshold|Minimum local-review detector confidence, within [0, 1].
crop_expansion|Positive expansion of the local-review crop relative to its proposed face box.
geometry_refinement.enabled|Refine propagated tracking geometry using a compatible local detector box.
geometry_refinement.min_local_confidence|Minimum local detector confidence accepted for a geometry refinement.
geometry_refinement.max_area_ratio|Maximum local-to-propagated box area ratio accepted for refinement.
geometry_refinement.max_center_distance|Maximum normalized local-to-propagated center distance accepted for refinement.
geometry_refinement.measurement_filter.enabled|Filter local geometry updates instead of applying every local detection directly.
geometry_refinement.measurement_filter.scope|Select all or tracking_only geometry updates; Base limits filtering to tracking_only frames.
geometry_refinement.measurement_filter.confidence_low|Lower detector-confidence endpoint used to interpolate measurement gains.
geometry_refinement.measurement_filter.confidence_high|Upper detector-confidence endpoint used to interpolate measurement gains.
geometry_refinement.measurement_filter.center_gain_low|Center-update gain at the low-confidence endpoint.
geometry_refinement.measurement_filter.center_gain_high|Center-update gain at the high-confidence endpoint.
geometry_refinement.measurement_filter.recovery_center_gain|Center-update gain for an anchor-recovery measurement.
geometry_refinement.measurement_filter.size_gain_low|Width/height update gain at the low-confidence endpoint.
geometry_refinement.measurement_filter.size_gain_high|Width/height update gain at the high-confidence endpoint.
geometry_refinement.measurement_filter.max_center_step|Maximum allowed normalized center update in one filtered measurement.
geometry_refinement.measurement_filter.max_size_ratio_per_update|Maximum width/height ratio change in one filtered measurement.
geometry_refinement.anchor_recovery.enabled|Retry failed local review around the nearest detector anchor to recover from optical-flow drift.
geometry_refinement.anchor_recovery.candidate_selection|Choose confidence or target_geometry ranking for anchor-recovery candidates; Base favors continuity with tracked geometry.
geometry_refinement.anchor_recovery.min_local_confidence|Minimum local confidence for the anchor-recovery candidate.
geometry_refinement.anchor_recovery.min_iou|Minimum overlap with tracked geometry for anchor recovery.
geometry_refinement.anchor_recovery.min_containment|Minimum containment agreement with tracked geometry for anchor recovery.
geometry_refinement.anchor_recovery.max_center_distance|Maximum normalized center separation from tracked geometry for anchor recovery.
edge_fallback.enabled|Try one translated local crop when the centered review fails and its square exceeds an image edge.
edge_fallback.shift_fraction|Fraction of the local crop used for the extra edge-directed translation.
match_min_iou|Minimum IoU for matching local review to the proposed face box.
match_min_containment|Minimum containment for matching local review to the proposed face box.
match_max_center_distance|Maximum normalized center distance for matching local review to the proposed face box.
match_max_area_ratio|Maximum local/proposed box area ratio for a local-review match.
passes|Optional ordered local-review cascade; absent or null uses shared input_size/crop_expansion, while an array must be nonempty with valid named entries.
""",
)
_register(
    "revalidation.policy.rule_gate",
    """
min_detector_frames|Minimum detector-supported frame count for ordinary track admission; positive.
min_local_match_fraction|Minimum local-review match fraction for ordinary track admission, within [0, 1].
min_local_confidence_fraction_gte_035|Minimum fraction of the trajectory with local detector confidence at least 0.35, within [0, 1].
min_joint_strong_anchor|Minimum continuous joint local-detector/Verifier anchor strength, within [0, 1].
strong_joint_anchor|Stronger joint anchor that can replace trajectory-wide Verifier support; at least min_joint_strong_anchor and at most 1.
min_verifier_pass_fraction|Minimum Verifier-supported fraction for ordinary track admission, within [0, 1].
short_track.enabled|Enable the detector-supported short-track admission path for tracks below ordinary persistence.
short_track.min_detector_frames|Minimum detector-supported frame count in the short-track path; positive.
short_track.max_detector_frames|Maximum detector-supported frame count in the short-track path; at least its minimum and below ordinary min_detector_frames.
short_track.min_local_match_fraction|Minimum local-review match fraction for short tracks, within [0, 1].
short_track.moderate_local_confidence_p50|Median local detector confidence for the moderate short-track conjunction, within [0, 1].
short_track.moderate_verifier_p50|Median Verifier score for the moderate short-track conjunction, within [0, 1].
short_track.strong_local_confidence_p50|Median local detector confidence for the strong conjunction; at least the moderate threshold and at most 1.
short_track.strong_verifier_p50|Median Verifier score for the strong conjunction; at least the moderate threshold and at most 1.
video_start_short_track.enabled|Enable the strict short-track exception only at the true frame-zero video boundary.
video_start_short_track.min_detector_frames|Required consecutive leading detector frames; within 1 and short_track.max_detector_frames.
video_start_short_track.min_detector_confidence_p50|Minimum median full-frame detector confidence for the video-start exception, within [0, 1].
video_start_short_track.min_local_match_fraction|Minimum local-review coverage for the video-start exception, within [0, 1].
video_start_short_track.min_local_confidence_p50|Minimum median local detector confidence for the video-start exception, within [0, 1].
strong_anchor_window_frames|Length of the continuous joint-evidence window in detector frames; at least 2. The weakest frame limits the window score.
normalizers.local_confidence_low|Low endpoint for normalizing local detector confidence before joint-anchor scoring.
normalizers.local_confidence_high|High endpoint for normalizing local detector confidence before joint-anchor scoring.
normalizers.verifier_score_low|Low endpoint for normalizing the Verifier score before joint-anchor scoring.
normalizers.verifier_score_high|High endpoint for normalizing the Verifier score before joint-anchor scoring.
""",
)
_register(
    "revalidation.policy.continuity",
    """
segment_max_center_jump|Maximum normalized center jump before splitting joint-evidence continuity; positive.
segment_max_area_ratio|Maximum area ratio before splitting joint-evidence continuity; at least 1.
""",
)
_register(
    "render",
    """
redaction.method|Apply Gaussian blur or mosaic to face regions; may be changed when rendering existing analysis.
redaction.box_scale|Scale each finalized face box around its center; 1 keeps coverage, above 1 expands it, and below 1 shrinks it.
redaction.gaussian.algorithm|pyramid approximates the full relative blur on a smaller working image; exact applies the full-resolution Gaussian path.
redaction.gaussian.max_side|Maximum side of the Gaussian pyramid working image, in pixels; positive integer.
redaction.gaussian.kernel_ratio|Blur-kernel size relative to the redacted region; positive.
redaction.gaussian.min_kernel|Minimum Gaussian kernel side, in pixels; positive odd integer.
redaction.gaussian.sigma|Nonnegative Gaussian sigma; 0 lets OpenCV derive it from the kernel.
redaction.mosaic.block_size_ratio|Mosaic block size relative to the larger region dimension; within (0, 1].
redaction.mosaic.min_block_size|Minimum mosaic block side in pixels; positive.
redaction.feather.enabled|Blend redaction opacity at region edges; false keeps opaque edges. Enabling requires ratio and min_pixels.
redaction.feather.ratio|Optional feather width relative to the smaller region dimension; required within [0, 0.5] when enabled.
redaction.feather.min_pixels|Optional minimum feather width in pixels; required positive when enabled.
video_output.backend|Use the in-process PyAV writer or an external FFmpeg executable.
video_output.encoder|Encoder name supported by the selected backend and installation; verify with command --dry-run.
video_output.pixel_format|Encoder pixel format such as yuv420p; compatibility depends on the encoder and frame geometry.
video_output.preset|Encoder-specific speed/compression preset; unsupported presets are rejected during codec preflight.
video_output.rate_control.mode|Select crf/cq quality control or vbr/cbr bitrate control; each mode permits a different set of sibling fields.
video_output.rate_control.quality|Quality control value within [0, 51] for crf/cq; cannot coexist with bitrate settings.
video_output.rate_control.bitrate|Target video bitrate for vbr/cbr, in bits per second; supports positive integer or decimal k/m/g suffixes.
video_output.rate_control.max_bitrate|Optional maximum video bitrate in vbr mode; supports positive integer or decimal k/m/g suffixes.
video_output.rate_control.buffer_size|Optional encoder rate-control buffer in cbr mode; supports positive integer or decimal k/m/g suffixes.
video_output.keyframe_interval|Explicit maximum GOP/keyframe interval in frames; 0 omits the override.
video_output.faststart|Request MP4 fast-start metadata placement; the current PyAV audio-remux path always enables faststart, so false does not disable it for that path.
video_output.audio.redacted|none removes audio; copy remuxes audio; aac requests AAC. PyAV can remux existing AAC but does not transcode other audio to AAC.
video_output.audio.bitrate|Target AAC audio bitrate used by FFmpeg transcoding; remuxed audio retains its original encoding and bitrate.
box_stabilization.enabled|Stabilize final exported face boxes over time before redaction.
box_stabilization.median_window|Temporal median filter window for output box geometry, in frames.
box_stabilization.min_segment_frames|Minimum contiguous segment length eligible for temporal box stabilization, in frames.
box_stabilization.reset_gap_frames|Frame gap that resets the stabilization segment.
box_stabilization.center_alpha|Exponential smoothing weight for output box center updates.
box_stabilization.size_alpha|Exponential smoothing weight for output box width/height updates.
box_stabilization.max_center_innovation|Maximum center innovation admitted by the stabilization filter, normalized by box size.
box_stabilization.max_size_ratio|Maximum width/height ratio change admitted by the stabilization filter.
box_stabilization.detector_anchor_strength|Weight given to real detector anchors when stabilizing interpolated boxes.
box_stabilization.detector_anchor_min_gap_frames|Minimum detector-anchor gap for the stabilization anchor adjustment, in frames.
box_stabilization.detector_anchor_max_gap_frames|Maximum detector-anchor gap for the stabilization anchor adjustment, in frames.
box_stabilization.change_point_reset.enabled|Reset stabilization across a verified abrupt and persistent box-scale change.
box_stabilization.change_point_reset.window_frames|Number of neighboring video frames per side considered for a scale-change reset.
box_stabilization.change_point_reset.min_detector_frames_per_side|Minimum real detector anchors required on each side of a scale-change reset.
box_stabilization.change_point_reset.min_instantaneous_scale_ratio|Minimum immediate box-scale ratio needed to propose a stabilization reset.
box_stabilization.change_point_reset.min_persistent_scale_ratio|Minimum sustained box-scale ratio across the reset window.
box_stabilization.change_point_reset.max_within_regime_ratio|Maximum size variation allowed within either side of the proposed reset.
box_stabilization.change_point_reset.min_scene_mean_absdiff|Minimum scene appearance difference supporting the scale-change reset.
""",
)

_SECTION_NOTES = {
    "output": "Retention controls affect diagnostic files, not the intended privacy policy. Audit/debug evidence is for investigation and increases artifact I/O.",
    "models": "Package task files, hashes, normalization, and resolved model paths come from the manifest; they are not user override fields. The selected model root is authoritative.",
    "runtime": "Provider and codec availability depend on the installation. Use doctor and the selected command's --dry-run before execution; these checks do not download models or create inference sessions.",
    "recognition": "Choose all to blur everyone, blur_only to blur people in reference photos, or exempt to keep people in those photos visible. The two photo modes require reference_dir and use JPG/JPEG/PNG/WebP images directly inside that folder (case-insensitive extensions); person names and subfolders are unnecessary. Hidden files and symlinks are skipped, and byte-identical photo copies are deduplicated. Different photos of the same person remain valid references. Only the largest detected face in each photo is considered, with no fallback to smaller faces. Multi-face selection, skipped-photo reasons, and import totals are logged on stderr; no usable reference faces is an error. With unknown_action=auto, uncertain or unmatched people remain visible in blur_only and are blurred in exempt. Model loading or inference errors stop processing. A higher similarity threshold is not the only recognition gate.",
    "scan": "Angles are degrees. Input sizes are positive multiples of 32. Detector-view arrays must be nonempty with unique names. Candidate-filter bounds apply when enabled. Scene-cut flow distances use the downscaled grayscale signature, not source-resolution pixels.",
    "streaming": "Frame limits count video frames and time limits use seconds of source video. max_missed_seconds must be positive, and max_retroactive_seconds must cover it. recent_frame_cache_frames accepts a nonnegative integer or null; the byte limit is nonnegative and remains a hard bound even with an automatic frame target. max_corridor_side_pixels must be at least 384. pre_roll_decode_chunk_frames must be positive and exceed tracking.kalman_optical_flow.max_coast_frames.",
    "tracking": "Most center-distance gates use center separation divided by the square root of the smaller box area (dimensionless). ROI flow residuals use working-image pixels. Scan-gap counts and video-frame counts are distinct. roi_size must be positive, roi_expansion must exceed 1, and max_source_canvas_side_pixels must cover roi_size. Enabled endpoint_affine_repair.max_frames must lie between 1 and tracking.endpoint_extension. Enabled fragment stitching requires at least two overlap/agreement frames, a nonnegative interval gap, and confidence/IoU/agreement fractions within [0,1].",
    "revalidation": "Local SCRFD and Verifier evidence control admission and geometry recovery. Short-track maximum detector count must remain below the ordinary minimum; strong thresholds must not be below moderate thresholds. Detector counts are evidence counts, not wall-clock FPS. Measurement-filter confidence endpoints must increase within [0,1]; gains are within [0,1] with high gains no smaller than low gains, max_center_step is positive, and max_size_ratio_per_update is at least 1. Anchor-recovery confidence, IoU, containment, and center-distance limits lie within [0,1].",
    "render": "Rendering can reuse an existing result JSON without inference. Encoder-specific options are checked by --dry-run. Audio remux preserves source audio; PyAV aac accepts existing AAC only, while FFmpeg can transcode and use audio.bitrate. Box stabilization requires an odd median_window of at least 3, min_segment_frames covering that window, positive reset_gap_frames, center/size alpha in (0,1], max_size_ratio at least 1, detector anchor strength in [0,1], and positive increasing anchor gaps. An enabled change-point reset needs a window of at least 2, detector counts fitting that window, scale ratios above 1, and nonnegative appearance difference.",
}


def normalize_path(path: str) -> str:
    return ".".join("*" if part.isdecimal() else part for part in path.split("."))


def description_for(path: str, spec: Mapping[str, Any]) -> str:
    description = _DESCRIPTIONS.get(normalize_path(path))
    if description is None:
        description = spec.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Missing configuration reference description: {path}")
    return " ".join(description.split())


def _cell(value: str) -> str:
    return html.escape(value, quote=False).replace("|", "\\|").replace("\n", " ")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def default_text(spec: Mapping[str, Any]) -> str:
    return (
        f"`{_cell(_json(spec['default']))}`"
        if spec["has_default"]
        else "Not set (optional)"
    )


def type_text(spec: Mapping[str, Any]) -> str:
    value = spec["type"]
    return " or ".join(value) if isinstance(value, list) else str(value)


def _constraints(spec: Mapping[str, Any]) -> str:
    values = []
    if "enum" in spec:
        values.append("Values: " + ", ".join(f"`{_json(v)}`" for v in spec["enum"]))
    for key, label in (
        ("minimum", "Minimum"),
        ("maximum", "Maximum"),
        ("exclusive_minimum", "Must exceed"),
        ("maximum_bps", "Maximum parsed bits/s"),
        ("maximum_bits", "Maximum parsed bits"),
    ):
        if key in spec:
            values.append(f"{label}: {spec[key]}")
    if spec.get("nullable"):
        values.append("Explicit null accepted")
    if "unit" in spec:
        values.append("Unit: " + str(spec["unit"]).replace("_", " "))
    return "; ".join(values)


def _full_contract() -> dict[str, Any]:
    from .cli_contract import _full_config_contract

    return _full_config_contract()


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    # Absolute installed paths must never make checked documentation machine-specific.
    stable = {
        key: contract[key]
        for key in ("schema_version", "schema", "defaults", "dotted_options")
    }
    return hashlib.sha256(_json(stable).encode("utf-8")).hexdigest()


def generate_reference(contract: Mapping[str, Any] | None = None) -> str:
    contract = _full_contract() if contract is None else contract
    options = contract["dotted_options"]
    lines = [
        "# PrivateFrame complete configuration reference",
        "",
        "This bundled reference is available offline after installation. It covers every current public YAML/CLI override, including advanced settings omitted from the curated `describe` response. Advanced settings remain supported; omission from discovery does not remove their behavior.",
        "",
        f"Public override paths: **{len(options)}**. Configuration schema version: **{contract['schema_version']}**.",
        "",
        f"<!-- configuration-contract-sha256: {contract_fingerprint(contract)} -->",
        "",
        "Generated by `python -m insightface.app.privateframe.config_reference`; do not hand-edit. Check for drift with `python -m insightface.app.privateframe.config_reference --check`. Use `--output PATH` to write a separate copy when the installed package is read-only.",
        "",
        "## Loading and overriding configuration",
        "",
        "Analysis starts from bundled `configs/base.yaml`. A custom `--config PATH` is a YAML mapping with `schema_version: 1`; its fields overlay the bundled Base unless `base_config` selects another base YAML file. `base_config` is a nonempty path, resolved relative to the overlay file when relative. It selects the immediate base document; it is not a recursive chain of overlays. `schema_version` and `base_config` are document fields, not dotted CLI options.",
        "",
        "Precedence: selected base YAML → custom YAML overlay → CLI dotted overrides. For rendering, the result JSON's `render_defaults` are overlaid by `--render-config` and then CLI `render.*` overrides. Render accepts only render settings; analyze/process and doctor accept the public configuration fields described below.",
        "",
        "Mappings merge recursively. Arrays supplied in YAML replace the entire array rather than merging elements. A CLI array override also replaces the array; indexed CLI overrides modify an existing element and cannot append past its length. Supply an array in custom YAML before using indexed CLI overrides, or pass one whole-array CLI override. Duplicate paths and overlapping parent/child CLI paths are rejected, so replacing an array and overriding one of its indices in the same command is invalid. The `rate_control` mapping is replaced as a unit only when the whole mapping is supplied; individual leaf overrides retain other sibling values.",
        "",
        "Dotted syntax is `--section.field VALUE` or `--section.field=VALUE`; values use YAML scalar/array syntax. Quote arrays and strings as required by your shell. Boolean values are `true`/`false`; `null` is accepted only where documented. Unknown keys and invalid types are rejected. `command --dry-run` is authoritative for cross-field and encoder-specific checks.",
        "",
        "```yaml",
        "schema_version: 1",
        "scan:",
        "  max_analysis_fps: 15",
        "render:",
        "  redaction:",
        "    method: mosaic",
        "```",
        "",
        "```sh",
        "insightface-privateframe process --input video.mp4 --output-dir output --config overlay.yaml --dry-run",
        "insightface-privateframe process --input video.mp4 --output-dir output --recognition.mode exempt --recognition.reference_dir reference_photos --dry-run",
        "insightface-privateframe process --input video.mp4 --output-dir output --recognition.mode blur_only --recognition.reference_dir reference_photos --dry-run",
        "```",
        "",
        "A table default is the literal packaged Base value, not an environment-resolved path or a measured runtime value. **Not set (optional)** differs from an explicit `null`: the field is absent from Base and its contextual behavior is explained in the description. `yaml` means no static leaf type is declared by the generated contract; the accompanying semantic requirements and dry-run still apply. Numeric ratios are dimensionless unless a different unit is stated. IoU is intersection over union; p50 is a median.",
        "",
        "## Optional arrays and cross-field rules",
        "",
        "`revalidation.passes` may be omitted or `null` to use shared revalidation settings. Otherwise it is a nonempty array of mappings with `name`, `input_size`, and `crop_expansion`. Names become nonempty unique text; input_size is a positive integer multiple of 32, and crop_expansion is a positive finite number. Indexed paths such as `revalidation.passes.0.input_size` work when the selected/overlay YAML already supplies those entries. To create the array from the CLI, supply a whole-array override without any indexed descendant override in that command. These paths are not listed as current Base indices because Base has no such array entries.",
        "",
        "`scan.passes` entries may contain optional `candidate_filter` settings. An enabled filter requires area fraction in [0,1], positive minimum height/width ratio, and an optional maximum ratio no smaller than that minimum. The aspect-ratio exemption area fraction must lie between the minimum area fraction and 1.",
        "",
        "For `render.video_output.rate_control`, crf/cq require exactly mode and quality; vbr requires mode and bitrate and optionally max_bitrate; cbr requires mode and bitrate and optionally buffer_size. Bitrate values are positive integers or strings with decimal k/m/g multipliers (1000/1000000/1000000000), bounded by signed 64-bit integer capacity after parsing. Changing only the mode leaf to vbr and adding bitrate leaves Base's quality value in place and is invalid. Replace the whole mapping when switching modes, as below. A quality-only leaf override is valid while staying in crf/cq.",
        "",
        "```sh",
        "insightface-privateframe process --input video.mp4 --output-dir output --render.video_output.rate_control '{mode: vbr, bitrate: 4M}' --dry-run",
        "insightface-privateframe process --input video.mp4 --output-dir output --render.video_output.rate_control.quality 22 --dry-run",
        "```",
        "",
        "Enabling feathering requires explicit `render.redaction.feather.ratio` in [0,0.5] and positive `min_pixels`. These fields have no Base default. Shrinking box_scale or blending redaction edges reduces covered/opaque pixels; inspect representative results when changing coverage.",
        "",
    ]
    section = None
    for path, spec in options.items():
        top = path.split(".", 1)[0]
        if top != section:
            section = top
            lines.extend(
                [
                    "",
                    f"## {top}",
                    "",
                    _SECTION_NOTES[top],
                    "",
                    "| Override path | Type | Base default | Meaning and constraints |",
                    "|---|---|---|---|",
                ]
            )
        description = description_for(path, spec)
        constraints = _constraints(spec)
        if constraints:
            description += " " + constraints + "."
        lines.append(
            f"| `{path}` | {_cell(type_text(spec))} | {default_text(spec)} | {_cell(description)} |"
        )
    lines.extend(
        [
            "",
            "## Compatibility-only diagnostics",
            "",
            "These existing diagnostic controls are intentionally excluded from curated discovery and the public override count. They are retained for compatibility and are not recommended for ordinary privacy-redaction workflows:",
            "",
            "| Control | Default | Purpose |",
            "|---|---|---|",
            "| `--debug PATH` on render/process | Not set | Additional annotated diagnostic video; requires an explicit path and is separate from the redacted output. |",
            "| `render.debug_line_thickness` | `2` | Diagnostic overlay line width in pixels. |",
            '| `render.video_output.audio.debug` | `"none"` | Audio policy for the diagnostic video; accepts none/copy/aac with the same backend limitations. |',
            "",
            "Runtime-computed fields such as `runtime.providers`, `runtime.resolved_provider`, model task file paths/preprocessing, and `models.manifest_path` are not user configuration controls and are not part of this reference.",
            "",
            "## Maintenance and validation sources",
            "",
            "This file is generated from the complete configuration contract and a reviewed description registry. The configuration tests compare its entire contents to a fresh generation so changes in public paths, defaults, types, or metadata require regeneration. Descriptions and cross-rules follow `configs/base.yaml`, `base_config.py`, `config.py`, `streaming.py`, `scene_cut.py`, `tracker.py`, `artifact_render.py`, and `box_stabilization.py`. The installed reference needs no network, model files, or video input to read or check.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the bundled reference without writing",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="reference file to generate or check",
    )
    args = parser.parse_args(argv)
    expected = generate_reference()
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Configuration reference is missing: {args.output}", file=sys.stderr)
            return 1
        if current != expected:
            print(
                f"Configuration reference is stale: {args.output}; regenerate with python -m insightface.app.privateframe.config_reference",
                file=sys.stderr,
            )
            return 1
        print(f"Configuration reference is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(f"Wrote configuration reference: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
