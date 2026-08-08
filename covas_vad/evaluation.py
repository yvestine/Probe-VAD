"""Frame-level evaluation for COVAS-VAD score JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

from covas_vad.video_record import VideoRecord


def temporal_testing_annotations(path: Path) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                annotations[parts[0].removesuffix(".mp4")] = parts[2:]
    return annotations


def get_video_labels(
    video: VideoRecord, annotations: dict[str, list[str]], normal_label: int
) -> list[int]:
    name = Path(video.path).name.removesuffix(".mp4")
    values = [value for value in annotations[name] if value != "-1"]
    starts, stops = values[::2], values[1::2]
    labels = list(video.label)
    if labels and len(labels) < len(starts):
        labels.extend([labels[-1]] * (len(starts) - len(labels)))

    output: list[int] = []
    for relative_frame in range(video.num_frames):
        absolute_frame = relative_frame + video.start_frame
        frame_label = normal_label
        interval_labels = labels if len(labels) > 1 else labels * len(starts)
        for start, stop, label in zip(starts, stops, interval_labels):
            if int(start) <= absolute_frame <= int(stop):
                frame_label = label
                break
        output.append(frame_label)
    return output


def expand_clip_scores(
    score_dict: dict[str, float],
    frame_interval: int,
    smoothing_sigma: float,
    smooth: bool,
) -> np.ndarray:
    ordered = [
        float(score)
        for _, score in sorted(score_dict.items(), key=lambda item: int(item[0]))
    ]
    clip_scores = np.asarray(ordered, dtype=np.float64)
    if smooth:
        clip_scores = gaussian_filter1d(clip_scores, sigma=smoothing_sigma)
    return np.repeat(clip_scores, frame_interval)


def save_metric(output_dir: Path, name: str, value: float) -> None:
    (output_dir / f"{name}.txt").write_text(f"{value}\n")


def main(args: argparse.Namespace) -> None:
    scores_dir = Path(args.scores_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = temporal_testing_annotations(Path(args.temporal_annotation_file))
    with Path(args.annotationfile_path).open() as handle:
        videos = [
            VideoRecord(line.strip().split(), args.root_path)
            for line in handle
            if line.strip()
        ]

    flat_scores: list[float] = []
    flat_labels: list[int] = []
    missing: list[str] = []
    for video in tqdm(videos, desc="evaluate"):
        name = Path(video.path).name.removesuffix(".mp4")
        if name.startswith(("abnormal_scene", "normal_scene")):
            name += ".mp4"
        score_path = scores_dir / f"{name}.json"
        if not score_path.is_file():
            missing.append(str(score_path))
            continue
        score_dict = json.loads(score_path.read_text())
        scores = expand_clip_scores(
            score_dict,
            args.frame_interval,
            args.smoothing_sigma,
            not args.no_smoothing,
        )[: video.num_frames]
        labels = get_video_labels(video, annotations, args.normal_label)
        if len(scores) < len(labels):
            scores = np.pad(scores, (0, len(labels) - len(scores)))
        flat_scores.extend(scores)
        flat_labels.extend(labels)

    if missing:
        print(f"WARNING: skipped {len(missing)} videos with missing score JSON")
    if not flat_scores:
        raise RuntimeError("no score files were evaluated")

    scores_array = np.asarray(flat_scores)
    binary_labels = np.asarray(flat_labels) != args.normal_label
    fpr, tpr, roc_thresholds = roc_curve(binary_labels, scores_array)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, pr_thresholds = precision_recall_curve(
        binary_labels, scores_array
    )
    pr_auc = float(auc(recall, precision))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    max_f1 = float(np.max(f1))

    roc_index = int(np.argmax(tpr - fpr))
    f1_index = int(np.argmax(f1))
    roc_threshold = float(roc_thresholds[roc_index])
    pr_threshold = float(pr_thresholds[min(f1_index, len(pr_thresholds) - 1)])

    save_metric(output_dir, "roc_auc", roc_auc)
    save_metric(output_dir, "pr_auc", pr_auc)
    save_metric(output_dir, "max_f1", max_f1)
    (output_dir / "optimal_thresholds.txt").write_text(
        f"ROC_Youden_J: {roc_threshold}\nPR_Max_F1: {pr_threshold}\n"
    )
    print(f"evaluated videos: {len(videos) - len(missing)}/{len(videos)}")
    print(f"ROC-AUC: {roc_auc:.8f}")
    print(f"PR-AUC:  {pr_auc:.8f}")
    print(f"Max-F1:  {max_f1:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--annotationfile_path", required=True)
    parser.add_argument("--temporal_annotation_file", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int, required=True)
    parser.add_argument("--no_smoothing", action="store_true")
    parser.add_argument("--smoothing_sigma", type=float, default=10.0)
    return parser.parse_args()


def cli_main() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli_main()
