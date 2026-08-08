import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.data.video_record import VideoRecord
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d

def temporal_testing_annotations(temporal_annotation_file):
    annotations = {}

    with open(temporal_annotation_file) as annotations_f:
        for line in annotations_f:
            parts = line.strip().split()
            video_name = str(parts[0]).replace(".mp4", "")
            annotation_values = parts[2:]
            annotations[video_name] = annotation_values

    return annotations


def get_video_labels(video_record, annotations, normal_label):
    video_name = Path(video_record.path).name
    video_name = video_name.replace(".mp4", "")
    labels = []
    # print(annotations.keys())
    video_annotations = [x for x in annotations[video_name] if x != "-1"]

    # Separate start and stop indices
    start_indices = video_annotations[::2]
    stop_indices = video_annotations[1::2]
    for frame_index in range(video_record.num_frames):
        frame_label = normal_label

        # Check if the current frame index falls within any annotation range
        if len(video_record.label) == 1:
            for start_idx, end_idx, label in zip(
                start_indices, stop_indices, video_record.label * len(start_indices)
            ):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label
        else:
            video_labels = video_record.label
            if len(video_labels) < len(start_indices):
                last_label = [video_record.label[-1]] * (len(start_indices) - len(video_labels))
                video_labels.extend(last_label)

            for start_idx, end_idx, label in zip(start_indices, stop_indices, video_labels):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label

        labels.append(frame_label)
    return labels


def calculate_weighted_scores(
    scores_dict, frame_interval, smooth=True, smoothing_sigma=10.0
):
    scores = []
    for frame_idx in scores_dict.keys():
        scores.append(scores_dict[frame_idx])
    if smooth:
        scores = gaussian_filter1d(scores, sigma=smoothing_sigma)
    scores = np.repeat(scores, frame_interval)
    return scores


def save_metric(output_dir, metric_name, metric_value):
    with open(output_dir / f"{metric_name}.txt", "w") as f:
        f.write(f"{metric_value}\n")
        

def main(
    root_path,
    annotationfile_path,
    temporal_annotation_file,
    scores_dir,
    captions_dir,
    output_dir,
    frame_interval,
    normal_label,
    without_labels,
    visualize,
    video_fps,
    no_smoothing,
    smoothing_sigma,
):
    # Convert paths to Path objects
    scores_dir = Path(scores_dir)
    captions_dir = Path(captions_dir) if captions_dir else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the temporal annotations
    if not without_labels:
        annotations = temporal_testing_annotations(temporal_annotation_file)
    # print(annotations)
    # Load video records from the annotation file
    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    flat_scores = []
    flat_labels = []

    for video in tqdm(video_list):
        video_name = Path(video.path).name
        video_name = video_name.replace(".mp4", "")
        if video_name.startswith("abnormal_scene") or video_name.startswith("normal_scene"):
            video_name += ".mp4"
        video_scores_path = scores_dir / f"{video_name}.json"
        try:
            with open(video_scores_path) as f:
                video_scores_dict = json.load(f)
        except:
            print("no file")
            continue
        # Get video labels
        if without_labels:
            video_labels = []
        else:
            video_labels = get_video_labels(video, annotations, normal_label)
        video_scores = calculate_weighted_scores(
            video_scores_dict,
            frame_interval,
            smooth=not no_smoothing,
            smoothing_sigma=smoothing_sigma,
        )
        video_scores = video_scores[: video.num_frames]
        
        # Pad scores with zeros if shorter than labels
        if not without_labels and len(video_scores) < len(video_labels):
            padding_length = len(video_labels) - len(video_scores)
            video_scores = np.pad(video_scores, (0, padding_length), 'constant', constant_values=0)
            
        # Extend scores and labels
        flat_scores.extend(video_scores)
        if not without_labels:
            flat_labels.extend(video_labels)
        if visualize:
            from src.utils.vis_utils import visualize_video

            captions_path = captions_dir / f"{video_name}.json"
            with open(captions_path) as f:
                video_captions = json.load(f)
            visualize_video(
                video_name,
                video_labels,
                video_scores,
                video_captions,
                video.path,
                video_fps,
                output_dir / f"{video_name}.mp4",
                normal_label,
                "{:06d}.jpg",
                None,
            )
    
    flat_scores = np.array(flat_scores)
    if not without_labels:
        flat_labels = np.array(flat_labels)
        flat_binary_labels = flat_labels != normal_label
        
        # Compute ROC AUC score
        fpr, tpr, threshold = roc_curve(flat_binary_labels[:len(flat_scores)], flat_scores)
        roc_auc = auc(fpr, tpr)
        save_metric(output_dir, "roc_auc", roc_auc)
        print(roc_auc)
        # Compute precision-recall curve
        precision, recall, th = precision_recall_curve(flat_binary_labels, flat_scores)
        pr_auc = auc(recall, precision)
        save_metric(output_dir, "pr_auc", pr_auc)
        print(pr_auc)
        # --- Optimal thresholds (defined only for labeled evaluation). ---
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold_roc = threshold[optimal_idx]
        print(f"Optimal threshold (ROC Youden's J): {optimal_threshold_roc}")

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        max_f1 = float(np.max(f1_scores))
        print(max_f1)
        save_metric(output_dir, "max_f1", max_f1)
        optimal_idx_f1 = np.argmax(f1_scores)
        # precision/recall contain one more element than threshold.
        optimal_threshold_pr = th[min(optimal_idx_f1, len(th) - 1)]
        print(f"Optimal threshold (Max F1): {optimal_threshold_pr}")

        with open(output_dir / "optimal_thresholds.txt", "w") as f:
            f.write(f"ROC_Youden_J: {optimal_threshold_roc}\n")
            f.write(f"PR_Max_F1: {optimal_threshold_pr}\n")

def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--temporal_annotation_file", type=str)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument(
        "--captions_dir",
        type=str,
        default=None,
        help="Caption directory; only needed with --visualize.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    # Optional arguments with defaults
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int)

    parser.add_argument("--without_labels", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--no_smoothing",
        action="store_true",
        help="Use raw clip scores without Gaussian smoothing.",
    )
    parser.add_argument(
        "--smoothing_sigma",
        type=float,
        default=10.0,
        help="Gaussian smoothing sigma in clip-score steps (default: 10).",
    )
    parser.add_argument("--video_fps", type=float)

    args = parser.parse_args()
    if args.temporal_annotation_file is None and not args.without_labels:
        parser.error("--temporal_annotation_file is required when --without_labels is not used")
    if args.visualize:
        if args.captions_dir is None:
            parser.error("--captions_dir is required when --visualize is used")
        if args.video_fps is None:
            parser.error("--video_fps is required when --visualize is used")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        args.root_path,
        args.annotationfile_path,
        args.temporal_annotation_file,
        args.scores_dir,
        args.captions_dir,
        args.output_dir,
        args.frame_interval,
        args.normal_label,
        args.without_labels,
        args.visualize,
        args.video_fps,
        args.no_smoothing,
        args.smoothing_sigma,
    )
