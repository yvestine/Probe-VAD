"""State-Dynamic Evidence Encoding (SDEE) frame construction.

Each input window is divided into five temporal bins.  One center RGB state
frame and one first-order dynamic evidence image are produced per bin.  The
dynamic image combines globally compensated appearance change and residual
optical-flow magnitude.  No second-order acceleration is used in D1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image


def as_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        value = np.asarray(frame.convert("RGB"))
    else:
        if hasattr(frame, "detach"):
            value = frame.detach().cpu().numpy()
        else:
            value = np.asarray(frame)
        # VideoLLaMA3's processor returns decoded frames as CHW torch tensors,
        # whereas OpenCV and PIL use HWC arrays.
        if (
            value.ndim == 3
            and value.shape[0] in (3, 4)
            and value.shape[2] not in (3, 4)
        ):
            value = np.transpose(value, (1, 2, 0))
        if value.ndim == 2:
            value = cv2.cvtColor(value, cv2.COLOR_GRAY2RGB)
        elif value.ndim == 3 and value.shape[2] == 4:
            value = cv2.cvtColor(value, cv2.COLOR_RGBA2RGB)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected RGB frame, got shape {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        finite = value[np.isfinite(value)]
        maximum = float(finite.max()) if len(finite) else 0.0
        if maximum <= 1.0 + 1e-6:
            value = value * 255.0
        value = np.clip(value, 0.0, 255.0)
    return np.ascontiguousarray(value.astype(np.uint8, copy=False))


def _resize_like(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == reference.shape[:2]:
        return frame
    return cv2.resize(
        frame,
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


def compensate_to_reference(
    frame_rgb: np.ndarray,
    reference_rgb: np.ndarray,
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Warp a frame onto the state frame using robust global affine motion."""
    frame_rgb = _resize_like(frame_rgb, reference_rgb)
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    points_ref = cv2.goodFeaturesToTrack(
        reference_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if points_ref is None or len(points_ref) < 8:
        return frame_rgb, False, identity
    points_frame, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        frame_gray,
        points_ref,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if points_frame is None or status is None:
        return frame_rgb, False, identity
    valid = status.reshape(-1).astype(bool)
    source = points_frame.reshape(-1, 2)[valid]
    target = points_ref.reshape(-1, 2)[valid]
    if len(source) < 8:
        return frame_rgb, False, identity
    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None or not np.all(np.isfinite(transform)):
        return frame_rgb, False, identity
    if inliers is not None and int(inliers.sum()) < 6:
        return frame_rgb, False, identity
    aligned = cv2.warpAffine(
        frame_rgb,
        transform,
        (reference_rgb.shape[1], reference_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return aligned, True, transform.astype(np.float32)


def _robust_unit_scale(
    values: np.ndarray,
    percentile: float = 95.0,
    minimum_scale: float = 1e-6,
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.percentile(finite, percentile)), minimum_scale)
    if scale <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def first_order_motion_evidence(
    frames_rgb: Sequence[np.ndarray],
    state_index: int,
    align_global_motion: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create D1 evidence: residual displacement plus appearance change."""
    if not frames_rgb:
        raise ValueError("motion evidence requires at least one frame")
    state = as_rgb_array(frames_rgb[state_index])
    aligned: list[np.ndarray] = []
    alignment_successes = 0
    transforms: list[list[list[float]]] = []
    for index, raw in enumerate(frames_rgb):
        frame = _resize_like(as_rgb_array(raw), state)
        if align_global_motion and index != state_index:
            frame, success, transform = compensate_to_reference(frame, state)
            alignment_successes += int(success)
        else:
            success = index == state_index
            transform = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            )
        aligned.append(frame)
        transforms.append(transform.tolist())

    if len(aligned) < 2:
        zero = np.zeros_like(state)
        return zero, {
            "frames_in_bin": len(aligned),
            "alignment_attempts": 0,
            "alignment_successes": 0,
            "transforms": transforms,
            "mean_change": 0.0,
            "mean_displacement": 0.0,
        }

    change_maps: list[np.ndarray] = []
    displacement_maps: list[np.ndarray] = []
    for previous, current in zip(aligned[:-1], aligned[1:]):
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        change_maps.append(
            cv2.absdiff(previous_gray, current_gray).astype(np.float32) / 255.0
        )
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        displacement_maps.append(
            cv2.magnitude(flow[..., 0], flow[..., 1]).astype(np.float32)
        )

    # Temporal means make D1 less sensitive to one noisy frame pair. Robust
    # spatial scaling preserves short local peaks without dataset parameters.
    change = np.mean(change_maps, axis=0)
    displacement = np.mean(displacement_maps, axis=0)
    # Floors prevent tiny compression/sensor noise from being stretched into a
    # saturated "strong motion" map in otherwise static segments.
    change_unit = _robust_unit_scale(change, minimum_scale=20.0 / 255.0)
    displacement_unit = _robust_unit_scale(displacement, minimum_scale=2.0)
    overlap = np.sqrt(np.clip(change_unit * displacement_unit, 0.0, 1.0))
    evidence = np.stack(
        (
            change_unit,
            displacement_unit,
            overlap,
        ),
        axis=-1,
    )
    evidence = np.rint(evidence * 255.0).astype(np.uint8)
    return evidence, {
        "frames_in_bin": len(aligned),
        "alignment_attempts": max(0, len(aligned) - 1) if align_global_motion else 0,
        "alignment_successes": alignment_successes,
        "transforms": transforms,
        "mean_change": float(np.mean(change)),
        "mean_displacement": float(np.mean(displacement)),
    }


def construct_sdee_sequence(
    frames: Sequence[Any],
    timestamps: Sequence[float],
    window_start: float,
    window_end: float,
    temporal_bins: int = 5,
    align_global_motion: bool = True,
    shuffle_motion: bool = False,
) -> tuple[list[Image.Image], list[float], dict[str, Any]]:
    """Return S1,M1,...,S5,M5 as ten PIL images."""
    if len(frames) != len(timestamps) or not frames:
        raise ValueError("frames and timestamps must be non-empty and aligned")
    if temporal_bins <= 0 or window_end <= window_start:
        raise ValueError("invalid SDEE temporal interval")
    pairs = sorted(
        zip((float(value) for value in timestamps), frames),
        key=lambda item: item[0],
    )
    sorted_times = np.asarray([item[0] for item in pairs], dtype=np.float64)
    sorted_frames = [as_rgb_array(item[1]) for item in pairs]
    edges = np.linspace(window_start, window_end, temporal_bins + 1)
    state_images: list[Image.Image] = []
    motion_images: list[Image.Image] = []
    midpoint_times: list[float] = []
    bin_details: list[dict[str, Any]] = []

    for bin_index in range(temporal_bins):
        start = float(edges[bin_index])
        end = float(edges[bin_index + 1])
        midpoint = (start + end) / 2.0
        if bin_index == temporal_bins - 1:
            indices = np.flatnonzero(
                (sorted_times >= start - 1e-9) & (sorted_times <= end + 1e-9)
            )
        else:
            indices = np.flatnonzero(
                (sorted_times >= start - 1e-9) & (sorted_times < end)
            )
        if not len(indices):
            indices = np.asarray(
                [int(np.argmin(np.abs(sorted_times - midpoint)))],
                dtype=np.int64,
            )
        local_times = sorted_times[indices]
        local_frames = [sorted_frames[int(index)] for index in indices]
        state_local_index = int(np.argmin(np.abs(local_times - midpoint)))
        state = local_frames[state_local_index]
        motion, motion_detail = first_order_motion_evidence(
            local_frames,
            state_local_index,
            align_global_motion=align_global_motion,
        )
        state_images.append(Image.fromarray(state, mode="RGB"))
        motion_images.append(Image.fromarray(motion, mode="RGB"))
        midpoint_times.append(midpoint)
        bin_details.append(
            {
                "bin": bin_index + 1,
                "interval": [start, end],
                "midpoint": midpoint,
                "candidate_timestamps": local_times.tolist(),
                "state_timestamp": float(local_times[state_local_index]),
                **motion_detail,
            }
        )

    motion_source_bins = list(range(temporal_bins))
    if shuffle_motion and temporal_bins > 1:
        # Fixed cyclic permutation is deterministic and guarantees that no
        # motion image remains paired with its original state image.
        motion_images = motion_images[1:] + motion_images[:1]
        motion_source_bins = list(range(1, temporal_bins)) + [0]

    sequence: list[Image.Image] = []
    sequence_times: list[float] = []
    for bin_index, (state, motion, midpoint) in enumerate(
        zip(state_images, motion_images, midpoint_times)
    ):
        sequence.extend((state, motion))
        sequence_times.extend((midpoint, midpoint + 1e-4))
    return sequence, sequence_times, {
        "window": [window_start, window_end],
        "temporal_bins": temporal_bins,
        "sequence_order": [
            label
            for index in range(1, temporal_bins + 1)
            for label in (f"S{index}", f"M{index}")
        ],
        "align_global_motion": align_global_motion,
        "shuffle_motion": shuffle_motion,
        "motion_source_bins": [index + 1 for index in motion_source_bins],
        "bins": bin_details,
    }


def decode_window_cv2(
    video_path: Path,
    start_time: float,
    end_time: float,
    fps: float,
) -> tuple[list[np.ndarray], list[float]]:
    """CPU decoder used only by the visualization preflight."""
    if fps <= 0 or end_time <= start_time:
        raise ValueError("invalid decode interval")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    targets = np.arange(start_time, end_time, 1.0 / fps, dtype=np.float64)
    if not len(targets):
        targets = np.asarray([(start_time + end_time) / 2.0])
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        for target in targets:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(target * 1000.0))
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            timestamps.append(float(target))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"decoder returned no frames for {video_path}")
    return frames, timestamps


def visualization_panel(
    sequence: Sequence[Image.Image],
    cell_width: int = 384,
) -> Image.Image:
    """Render five S/M rows for human inspection; labels are not model input."""
    if len(sequence) % 2:
        raise ValueError("SDEE visualization requires state/motion pairs")
    rows: list[np.ndarray] = []
    for pair_index in range(len(sequence) // 2):
        cells: list[np.ndarray] = []
        for offset, prefix in ((0, "S"), (1, "M")):
            image = as_rgb_array(sequence[pair_index * 2 + offset])
            scale = cell_width / image.shape[1]
            resized = cv2.resize(
                image,
                (cell_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            labeled = resized.copy()
            cv2.putText(
                labeled,
                f"{prefix}{pair_index + 1}",
                (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cells.append(labeled)
        height = min(cell.shape[0] for cell in cells)
        cells = [cell[:height] for cell in cells]
        rows.append(np.concatenate(cells, axis=1))
    width = max(row.shape[1] for row in rows)
    padded = [
        cv2.copyMakeBorder(
            row,
            0,
            0,
            0,
            width - row.shape[1],
            cv2.BORDER_CONSTANT,
        )
        for row in rows
    ]
    return Image.fromarray(np.concatenate(padded, axis=0), mode="RGB")
