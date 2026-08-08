import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from libs.llama.llama import Dialog, Llama
from src.data.video_record import VideoRecord
from src.utils.path_utils import find_unprocessed_videos
import torch
import random

DEFAULT_SEED = 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_suspicious_phrase(phrase: str) -> str:
    """
    Remove any leading/trailing square brackets and extra spaces.
    E.g. "['theft', 'burglary']" -> "'theft', 'burglary'"
    E.g. "[] " -> ""
    """
    phrase = phrase.strip()
    if phrase.startswith('['):
        phrase = phrase[1:]
    if phrase.endswith(']'):
        phrase = phrase[:-1]
    return phrase.strip()


class LLMAnomalyScorerLocalOptimal:
    def __init__(
        self,
        root_path,
        annotationfile_path,
        batch_size,
        frame_interval,
        context_prompt,
        format_prompt,
        output_scores_dir,
        captions_dir,
        ckpt_dir,
        tokenizer_path,
        temperature,
        top_p,
        max_seq_len,
        max_gen_len,
        threshold=None,
        suspicious_phrases_json=None,
        highest_lowest_json=None,  # <-- NEW: pass path to your JSON with highest_avg_score
        seed=DEFAULT_SEED,
    ):
        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.context_prompt = context_prompt
        self.format_prompt = format_prompt
        self.output_scores_dir = output_scores_dir
        self.captions_dir = captions_dir
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.threshold = threshold
        self.seed = seed

        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.batch_size,
            seed=self.seed,
        )

        # Load suspicious phrases, if provided
        self.suspicious_phrases = {}
        if suspicious_phrases_json is not None:
            with open(suspicious_phrases_json, "r") as f:
                loaded = json.load(f)
            new_susp = {}
            for old_key, val in loaded.items():
                base_name = old_key.split("/")[-1]  # e.g. "Abuse028_x264"
                new_susp[base_name] = val
            
            self.suspicious_phrases = new_susp

        self.highest_lowest_data = {}
        if highest_lowest_json is not None:
            with open(highest_lowest_json, "r") as f:
                self.highest_lowest_data = json.load(f)

    def _truncate_suspicious_text(self, base_prompt: str, suspicious_text: str, tail_prompt: str = "") -> str:
        tokenizer = self.generator.tokenizer

        base_tokens = tokenizer.encode(base_prompt, bos=False, eos=False)
        tail_tokens = tokenizer.encode(tail_prompt, bos=False, eos=False) if tail_prompt else []
        used_tokens = len(base_tokens) + len(tail_tokens)
        budget = self.max_seq_len - used_tokens
        if budget <= 0:
            return ""

        words = suspicious_text.split()
        truncated_text = ""
        for w in words:
            candidate = (truncated_text + " " + w).strip() if truncated_text else w
            candidate_tokens = tokenizer.encode(candidate, bos=False, eos=False)
            if len(candidate_tokens) > budget:
                break
            truncated_text = candidate

        return truncated_text.strip()

    def _prepare_dialogs(self, video_basename, captions, batch_frame_idxs):
        suspicious_raw = self.suspicious_phrases.get(video_basename, "").strip()
        suspicious_text = clean_suspicious_phrase(suspicious_raw)

        system_prompt = f"{self.context_prompt}"
        if (suspicious_text and suspicious_text.lower() != "none" and 
            suspicious_text != "" and "no " not in suspicious_text.lower() and "no_" not in suspicious_text.lower() and len(self.generator.tokenizer.encode(suspicious_text, bos=False, eos=False))<=self.max_seq_len):
            truncated_susp = self._truncate_suspicious_text(
                base_prompt=system_prompt + "\n[Potentially reported suspicious activities: ",
                suspicious_text=suspicious_text,
                tail_prompt="]\n\n" + self.format_prompt,
            )
            if truncated_susp:
                system_prompt += f"\n[Potentially reported suspicious activities: {truncated_susp}]"

        system_prompt += f"\n\n{self.format_prompt}"
        batch_clip_caption = [captions[str(idx)] for idx in batch_frame_idxs]



        # If suspicious_text was empty, we skip
        if "[Potentially reported suspicious activities:" not in system_prompt:
            print(f"Skipping {video_basename} because suspicious phrase was empty or invalid.")
            return None

        dialogs: List[Dialog] = []
        for clip_caption in batch_clip_caption:
            dialogs.append([
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": clip_caption},
            ])
        return dialogs


    def _parse_score(self, response: str) -> float:
        """
        e.g. "[0.3]" => 0.3
        """
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        return float(match.group(1)) if match else -1.0

    def _interpolate_unmatched_scores(self, scores):
        valid_scores = [(int(k), v) for k, v in scores.items() if v != -1]
        if not valid_scores:
            return scores

        valid_scores.sort(key=lambda x: x[0])
        xs = [x for x, _ in valid_scores]
        ys = [y for _, y in valid_scores]

        all_frames = sorted(int(k) for k in scores.keys())
        interpolated = np.interp(all_frames, xs, ys)

        final_scores = {}
        for frame_idx, val in zip(all_frames, interpolated):
            final_scores[str(frame_idx)] = round(float(val), 2)
        return final_scores

    def _score_temporal_summaries(self, video, temporal_summaries):
        video_scores = {}
        video_basename = Path(video.path).stem
        if video_basename.endswith(".mp4"):
            video_basename = video_basename.replace(".mp4", "")

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Scoring {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + self.batch_size * self.frame_interval,
                video.num_frames
            )
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)

            dialogs = self._prepare_dialogs(
                video_basename=video_basename,
                captions=temporal_summaries,
                batch_frame_idxs=batch_frame_idxs,
            )
            if dialogs is None:
                return None

            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            for result, frame_idx in zip(results, batch_frame_idxs):
                response_text = result["generation"]["content"].strip()
                score = self._parse_score(response_text)
                video_scores[str(frame_idx)] = score

        video_scores = self._interpolate_unmatched_scores(video_scores)
        return video_scores

    def process_video(self, video):
        video_name = Path(video.path).name
        video_basename = video_name.replace(".mp4", "")

        stats = self.highest_lowest_data.get(video_basename)
        threshold_margin = self.threshold

        if stats is None:
            if self.highest_lowest_data:
                print(f"Skipping {video_basename}, not found in highest_lowest_json.")
                return
        else:
            try:
                highest_avg_score = float(stats.get("highest_avg_score"))
            except (TypeError, ValueError):
                print(
                    f"Skipping {video_basename}, invalid highest_avg_score entry: {stats.get('highest_avg_score')}"
                )
                return

            if threshold_margin is None:
                std_value = stats.get("std")
                if std_value is not None:
                    try:
                        std_float = float(std_value)
                    except (TypeError, ValueError):
                        std_float = None
                    else:
                        threshold_margin = (abs(std_float)) ** 2

            if threshold_margin is not None:
                try:
                    threshold_margin = abs(float(threshold_margin))
                except (TypeError, ValueError):
                    print(f"Skipping {video_basename}, invalid threshold value: {threshold_margin}")
                    return
                lower = max(0.0, 0.5 - threshold_margin)
                upper = min(1.0, 0.5 + threshold_margin)
                if not (lower <= highest_avg_score <= upper):
                    print(
                        f"Skipping {video_basename}, highest_avg_score={highest_avg_score:.3f} outside [{lower:.3f}, {upper:.3f}]"
                    )
                    return

        if self.suspicious_phrases and video_basename not in self.suspicious_phrases:
            print(f"Skipping {video_basename}, because no suspicious phrase found.")
            return

        temporal_summaries_path = Path(self.captions_dir) / f"{video_name}.json"
        if not temporal_summaries_path.exists():
            print(f"Skipping {video_basename}, missing captions at {temporal_summaries_path}")
            return
        with open(temporal_summaries_path, "r") as f:
            temporal_summaries = json.load(f)

        video_scores = self._score_temporal_summaries(video, temporal_summaries)
        if video_scores is None:
            return
        output_path = Path(self.output_scores_dir) / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_scores, f, indent=4)


def run(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    context_prompt,
    format_prompt,
    output_scores_dir,
    captions_dir,
    ckpt_dir,
    tokenizer_path,
    temperature,
    top_p,
    max_seq_len,
    max_gen_len,
    resume,
    pathname,
    num_jobs,
    job_index,
    threshold=None,
    suspicious_phrases_json=None,
    highest_lowest_json=None,  
    seed=DEFAULT_SEED,
    
):
    set_seed(seed)

    # Save prompts for reproducibility
    Path(output_scores_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_scores_dir) / "context_prompt.txt", "w") as f:
        f.write(context_prompt)
    with open(Path(output_scores_dir) / "format_prompt.txt", "w") as f:
        f.write(format_prompt)

    # Read videos from annotation file
    with open(annotationfile_path, "r") as ann_f:
        video_list = [VideoRecord(x.strip().split(), root_path) for x in ann_f]
    # Possibly split for multiple jobs
    video_list = list(np.array_split(video_list, num_jobs)[job_index])

    # Possibly skip processed
    if resume:
        video_list = find_unprocessed_videos(
            video_list,
            output_scores_dir,
            pathname,
        )

    llm_scorer = LLMAnomalyScorerLocalOptimal(
        root_path=root_path,
        annotationfile_path=annotationfile_path,
        batch_size=batch_size,
        frame_interval=frame_interval,
        context_prompt=context_prompt,
        format_prompt=format_prompt,
        output_scores_dir=output_scores_dir,
        captions_dir=captions_dir,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
        threshold=threshold,
        suspicious_phrases_json=suspicious_phrases_json,
        highest_lowest_json=highest_lowest_json,  # pass the JSON
        seed=seed,
    )

    for video in tqdm(video_list):
        llm_scorer.process_video(video)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--context_prompt", type=str)
    parser.add_argument("--format_prompt", type=str)
    parser.add_argument("--output_scores_dir", type=str)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument(
        "--suspicious_phrases_json",
        type=str,
        default=None,
        help="JSON mapping videos -> suspicious crime strings."
    )
    parser.add_argument(
        "--highest_lowest_json",
        type=str,
        default=None,
        help="JSON with each video's highest/lowest intervals and highest_avg_score."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="margin threshold"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()


    if not (args.context_prompt and args.format_prompt and args.output_scores_dir):
        parser.error("Scoring requires --context_prompt, --format_prompt, and --output_scores_dir.")
    return args


if __name__ == "__main__":
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        context_prompt=args.context_prompt,
        format_prompt=args.format_prompt,
        output_scores_dir=args.output_scores_dir,
        captions_dir=args.captions_dir,
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len,
        resume=args.resume,
        pathname=args.pathname,
        num_jobs=args.num_jobs,
        job_index=args.job_index,
        threshold=args.threshold,
        suspicious_phrases_json=args.suspicious_phrases_json,
        highest_lowest_json=args.highest_lowest_json,  # pass to main run
        seed=args.seed,
    )
