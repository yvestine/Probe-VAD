"""Caption-only cumulative likelihood scoring.

This is an ablation counterpart to the video cumulative-threshold scorer.  It
reuses precomputed clip captions and scores the ten YES/NO threshold questions
with the local Llama 3.1 checkpoint.  No video, caption generation, tags,
refinement, or score fusion is performed here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from libs.llama.llama import Llama

LOGGER = logging.getLogger("caption_cumulative_likelihood")
THRESHOLDS = np.arange(0.1, 1.01, 0.1, dtype=np.float64)
YES_TOKEN = "YES"
NO_TOKEN = "NO"

SYSTEM_PROMPT = (
    "You are a careful video anomaly detector for public surveillance scenes. "
    "Judge only the behavior described in the current clip caption. A score "
    "of 0 means ordinary, safe activity and 1 means clearly abnormal, "
    "dangerous, violent, criminal, or emergency activity."
)


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def pava_decreasing(values: list[float]) -> list[float]:
    """Least-squares projection onto q[0] >= ... >= q[n-1]."""
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[0] / left[1] >= right[0] / right[1]:
                break
            blocks[-2] = [
                left[0] + right[0],
                left[1] + right[1],
            ]
            blocks.pop()
    output: list[float] = []
    for total, count in blocks:
        output.extend([total / count] * int(count))
    return output


def split_for_job(names: list[str], num_jobs: int, job_index: int) -> list[str]:
    return list(np.array_split(np.asarray(names, dtype=object), num_jobs)[job_index])


def load_names(index_file: Path) -> list[str]:
    return [line.split()[0].removesuffix(".mp4") for line in index_file.read_text().splitlines() if line.strip()]


def build_prompt_tokens(llama: Llama, caption: str, threshold: float) -> list[int]:
    question = (
        f"Caption of the current video clip:\n{caption.strip()}\n\n"
        f"On the severity scale defined by the system, is the anomaly severity "
        f"described in this clip at least {threshold:.1f}? "
        "Judge only the current caption. Answer exactly YES or NO."
    )
    dialog = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return llama.formatter.encode_dialog_prompt(dialog)


@torch.inference_mode()
def yes_no_loglikelihood(
    llama: Llama, prompt_tokens: list[int], temperature: float
) -> tuple[float, float, float]:
    """Score one threshold; YES and NO are verified to be single tokens."""
    tokenizer = llama.tokenizer
    yes_ids = tokenizer.encode(YES_TOKEN, bos=False, eos=False)
    no_ids = tokenizer.encode(NO_TOKEN, bos=False, eos=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(f"YES/NO must be single tokens, got {yes_ids}/{no_ids}")

    device = torch.device("cuda")
    prompt_len = len(prompt_tokens)
    tokens = torch.tensor(
        [prompt_tokens + [yes_ids[0]], prompt_tokens + [no_ids[0]]],
        dtype=torch.long,
        device=device,
    )
    logits = llama.model.forward(tokens, 0)
    last_prompt_logits = logits[:, prompt_len - 1, :].float()
    log_probs = torch.log_softmax(last_prompt_logits, dim=-1)
    yes_ll = float(log_probs[0, yes_ids[0]].item())
    no_ll = float(log_probs[1, no_ids[0]].item())
    margin = (yes_ll - no_ll) / temperature
    probability = 1.0 / (1.0 + math.exp(-margin))
    return yes_ll, no_ll, probability


def score_caption(llama: Llama, caption: str, temperature: float) -> tuple[float, dict[str, Any]]:
    raw: list[dict[str, float]] = []
    probabilities: list[float] = []
    for threshold in THRESHOLDS:
        prompt_tokens = build_prompt_tokens(llama, caption, float(threshold))
        yes_ll, no_ll, probability = yes_no_loglikelihood(
            llama, prompt_tokens, temperature
        )
        raw.append(
            {
                "threshold": float(threshold),
                "yes_loglik": yes_ll,
                "no_loglik": no_ll,
                "tail_probability": probability,
            }
        )
        probabilities.append(probability)
    adjusted = pava_decreasing(probabilities)
    return float(np.mean(adjusted)), {
        "thresholds": raw,
        "monotonic_tail_probabilities": adjusted,
        "score": float(np.mean(adjusted)),
    }


def process_video(args: argparse.Namespace, llama: Llama, name: str) -> None:
    caption_path = args.captions_dir / f"{name}.json"
    if not caption_path.exists():
        LOGGER.warning("Missing caption file: %s", caption_path)
        return
    captions = json.loads(caption_path.read_text())
    scores_path = args.output_dir / f"{name}.json"
    details_path = args.output_dir / "_threshold_details" / f"{name}.json"
    scores = json.loads(scores_path.read_text()) if args.resume and scores_path.exists() else {}
    details = json.loads(details_path.read_text()) if args.resume and details_path.exists() else {}
    for key in sorted(captions, key=lambda x: int(x)):
        if args.resume and key in scores and key in details:
            continue
        score, detail = score_caption(llama, str(captions[key]), args.temperature)
        scores[str(key)] = score
        details[str(key)] = detail
        atomic_write(scores_path, scores)
        atomic_write(details_path, details)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=Path, required=True)
    parser.add_argument("--captions_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("./libs/llama/llama3.1-8b"))
    parser.add_argument("--tokenizer_path", type=Path, default=Path("./libs/llama/llama3.1-8b/tokenizer.model"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=2)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--progress_label", default="Caption-E0")
    parser.add_argument("--progress_position", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_level", default="ERROR")
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    names = split_for_job(load_names(args.index_file), args.num_jobs, args.job_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    llama = Llama.build(
        ckpt_dir=str(args.ckpt_dir),
        tokenizer_path=str(args.tokenizer_path),
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        seed=1,
    )
    for name in tqdm(
        names,
        desc=args.progress_label,
        unit="video",
        position=args.progress_position,
        dynamic_ncols=True,
        leave=True,
    ):
        process_video(args, llama, name)


if __name__ == "__main__":
    main()
