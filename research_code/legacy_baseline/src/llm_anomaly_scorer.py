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



class LLMAnomalyScorer:
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
        seed,
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
        self.seed = seed

        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.batch_size,
            seed=self.seed,
        )

    def _prepare_dialogs(self, captions, batch_frame_idxs):
    
        prompt = self.context_prompt + " " + self.format_prompt
        batch_clip_caption = [f"{captions[str(idx)]}." for idx in batch_frame_idxs]
        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": clip_caption},
            ]
            for clip_caption in batch_clip_caption
        ]
        return dialogs

    def _parse_score(self, response):
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        score = float(match.group(1)) if match else -1
        return score

    def _interpolate_unmatched_scores(self, scores):
        valid_scores = [(idx, score) for idx, score in scores.items() if score != -1]
        video_scores = np.interp(list(scores.keys()), *zip(*valid_scores))

        return dict(zip(scores.keys(), video_scores))

    def _score_temporal_summaries(self, video, temporal_summaries):
        video_scores = {}

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)

            dialogs = self._prepare_dialogs(temporal_summaries, batch_frame_idxs)

            results = self.generator.chat_completion(
                dialogs,
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            for result, frame_idx in zip(results, batch_frame_idxs):
                response = result["generation"]["content"]
                # print(response)
                score = self._parse_score(response)
                # print(score)
                video_scores[str(frame_idx)] = score

        video_scores = self._interpolate_unmatched_scores(video_scores)

        return video_scores

    def process_video(self, video):
        video_name = Path(video.path).name
        video_name = video_name.replace(".mp4", "")
        # Score temporal summaries
        temporal_summaries_path = Path(self.captions_dir) / f"{video_name}.json"
        with open(temporal_summaries_path) as f:
            temporal_summaries = json.load(f)

        video_scores = self._score_temporal_summaries(video, temporal_summaries)

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
    seed,
):
    set_seed(seed)

    output_scores_dir = Path(output_scores_dir)
    output_scores_dir.mkdir(parents=True, exist_ok=True)
    with open(output_scores_dir / "context_prompt.txt", "w") as f:
        f.write(context_prompt)
    with open(output_scores_dir / "format_prompt.txt", "w") as f:
        f.write(format_prompt)

    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    video_list = list(np.array_split(video_list, num_jobs)[job_index])
    if resume:
        video_list = find_unprocessed_videos(
            video_list, output_scores_dir, pathname
        )

    llm_anomaly_scorer = LLMAnomalyScorer(
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
        seed=seed,
    )

    for video in video_list:
        llm_anomaly_scorer.process_video(video)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--context_prompt", type=str)
    parser.add_argument("--format_prompt", type=str)
    parser.add_argument("--output_scores_dir", type=str)
    parser.add_argument("--captions_dir", type=str)
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

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
        seed=args.seed,
    )
