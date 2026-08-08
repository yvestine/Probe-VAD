"""Utilities used exclusively by the COVAS-VAD E0 scoring pipeline."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.cache_utils import Cache, DynamicCache

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"
CUMULATIVE_THRESHOLDS = tuple(index / 10.0 for index in range(1, 11))


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration: float


@dataclass(frozen=True)
class VideoWindow:
    center_frame: int
    start_time: float
    end_time: float


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"invalid video metadata for {video_path}: fps={fps}, frames={frame_count}"
        )
    return VideoInfo(fps=fps, frame_count=frame_count, duration=frame_count / fps)


def iter_video_windows(
    info: VideoInfo, frame_interval: int = 16, window_seconds: float = 10.0
) -> Iterable[VideoWindow]:
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    for center_frame in range(0, info.frame_count, frame_interval):
        center_time = center_frame / info.fps
        yield VideoWindow(
            center_frame=center_frame,
            start_time=max(0.0, center_time - window_seconds / 2.0),
            end_time=min(info.duration, center_time + window_seconds / 2.0),
        )


def load_video_names(
    video_dir: Path, index_file: Path | None = None, video_list: Sequence[str] | None = None
) -> List[str]:
    if index_file is not None and video_list:
        raise ValueError("use either --index_file or --video_list, not both")
    if index_file is not None:
        with index_file.open() as handle:
            names = [line.split()[0] for line in handle if line.strip()]
    elif video_list:
        names = list(video_list)
    else:
        names = [
            str(path.relative_to(video_dir))
            for path in sorted(video_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ]
    return names


def resolve_video_path(video_dir: Path, name: str) -> Path:
    raw = Path(name)
    video_suffixes = (".mp4", ".avi", ".mov", ".mkv")
    nested = video_dir / raw
    candidates = [nested]
    if raw.suffix.lower() not in video_suffixes:
        # Dataset identifiers such as "A.Beautiful.Mind.2001__..." contain
        # dots that pathlib interprets as a suffix even though the annotation
        # intentionally omits the real video extension. Append instead of
        # replacing that apparent suffix.
        candidates.extend(Path(f"{nested}{suffix}") for suffix in video_suffixes)
    # UCF index entries may contain a category while videos are stored flat.
    flat = video_dir / raw.name
    candidates.append(flat)
    if raw.suffix.lower() not in video_suffixes:
        candidates.extend(Path(f"{flat}{suffix}") for suffix in video_suffixes)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"video not found for {name}; tried: {candidates}")


def output_stem(name: str) -> str:
    """Return the output basename, stripping only a real video extension.

    Dataset identifiers may contain dots without including a file extension,
    for example ``Spectre.2015__#..._label_A``.  ``Path.stem`` would truncate
    that identifier to ``Spectre`` and make multiple clips overwrite the same
    score JSON.  Keep such identifiers intact and remove only a recognized
    video suffix.
    """
    basename = Path(name).name
    path = Path(basename)
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return path.stem
    return basename


def split_for_job(items: Sequence[str], num_jobs: int, job_index: int) -> List[str]:
    if num_jobs < 1 or not 0 <= job_index < num_jobs:
        raise ValueError("job_index must be in [0, num_jobs)")
    return list(items[job_index::num_jobs])


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=4, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open() as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Ignoring unreadable checkpoint %s", path)
        return {}


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str | None = None,
    attn_implementation: str = "flash_attention_2",
):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    if not device.startswith("cuda") and attn_implementation == "flash_attention_2":
        LOGGER.warning(
            "flash_attention_2 requires CUDA; using eager attention on %s", device
        )
        attn_implementation = "eager"
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device,
        "torch_dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def prepare_inputs(model, processor, conversation) -> MutableMapping[str, Any]:
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = (
                value.to(device=device, dtype=dtype)
                if key == "pixel_values"
                else value.to(device)
            )
    return inputs


def _tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor does not expose a tokenizer")
    return tokenizer


@torch.inference_mode()
def candidate_log_likelihood(
    model,
    processor,
    base_inputs: Mapping[str, Any],
    candidate: str,
    length_normalize: bool,
) -> float:
    """Return conditional candidate log likelihood, supporting multi-token labels."""
    tokenizer = _tokenizer(processor)
    candidate_ids = tokenizer(
        candidate, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(base_inputs["input_ids"].device)
    if candidate_ids.numel() == 0:
        raise RuntimeError(f"candidate tokenized to an empty sequence: {candidate!r}")

    prompt_ids = base_inputs["input_ids"]
    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    attention_mask = torch.cat(
        [
            base_inputs.get("attention_mask", torch.ones_like(prompt_ids)),
            torch.ones_like(candidate_ids),
        ],
        dim=1,
    )
    forward_inputs = dict(base_inputs)
    forward_inputs.update(
        input_ids=input_ids, attention_mask=attention_mask
    )
    outputs = model(**forward_inputs, use_cache=False, return_dict=True)
    token_count = candidate_ids.shape[1]
    # Candidate tokens are the final text tokens. Visual placeholder compression
    # only changes positions earlier in the sequence, so the K logits immediately
    # preceding the final K tokens remain correctly aligned.
    candidate_logits = outputs.logits[:, -token_count - 1 : -1, :]
    if candidate_logits.shape[1] != token_count:
        raise RuntimeError(
            f"candidate/logit alignment failed: {token_count} tokens, "
            f"{candidate_logits.shape[1]} prediction positions"
        )
    token_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    selected = token_log_probs.gather(2, candidate_ids.unsqueeze(-1)).squeeze(-1)
    value = selected.mean() if length_normalize else selected.sum()
    return float(value.item())


def decreasing_isotonic_projection(values: Sequence[float]) -> List[float]:
    """L2-project values onto a non-increasing sequence using PAVA."""
    if not values:
        return []
    # Blocks are [mean, weight, start, end]. Adjacent violations are pooled.
    blocks: List[List[float | int]] = []
    for index, value in enumerate(values):
        blocks.append([float(value), 1, index, index])
        while len(blocks) >= 2 and float(blocks[-2][0]) < float(blocks[-1][0]):
            right = blocks.pop()
            left = blocks.pop()
            weight = int(left[1]) + int(right[1])
            mean = (
                float(left[0]) * int(left[1])
                + float(right[0]) * int(right[1])
            ) / weight
            blocks.append([mean, weight, int(left[2]), int(right[3])])
    projected = [0.0] * len(values)
    for mean, _, start, end in blocks:
        for index in range(int(start), int(end) + 1):
            projected[index] = float(mean)
    return projected


def cumulative_score_from_tail_probabilities(
    probabilities: Sequence[float], monotonic_projection: bool = True
) -> Tuple[float, List[float], int]:
    """Approximate E[S] from P(S >= tau) on the 0.1 threshold grid."""
    if len(probabilities) != len(CUMULATIVE_THRESHOLDS):
        raise ValueError(
            f"expected {len(CUMULATIVE_THRESHOLDS)} tail probabilities, "
            f"got {len(probabilities)}"
        )
    raw = [min(1.0, max(0.0, float(value))) for value in probabilities]
    violations = sum(raw[index] < raw[index + 1] for index in range(len(raw) - 1))
    adjusted = decreasing_isotonic_projection(raw) if monotonic_projection else raw
    # Grid spacing is 0.1, hence 0.1 * sum(p_k) == mean(p_k).
    score = float(sum(adjusted) / len(adjusted))
    return score, adjusted, violations


@torch.inference_mode()
def threshold_yes_no_likelihood(
    model,
    processor,
    conversation,
    yes_candidate: str = "YES",
    no_candidate: str = "NO",
    length_normalize: bool = False,
    temperature: float = 1.0,
) -> Tuple[float, float, float]:
    """Return P(YES), log P(YES sequence), and log P(NO sequence)."""
    if temperature <= 0:
        raise ValueError("likelihood temperature must be positive")
    inputs = prepare_inputs(model, processor, conversation)
    yes_ll = candidate_log_likelihood(
        model, processor, inputs, yes_candidate, length_normalize
    )
    no_ll = candidate_log_likelihood(
        model, processor, inputs, no_candidate, length_normalize
    )
    logits = torch.tensor([no_ll, yes_ll], dtype=torch.float64) / temperature
    probability = float(torch.softmax(logits, dim=0)[1].item())
    return probability, yes_ll, no_ll


def _move_processor_inputs(
    values: MutableMapping[str, Any], device: torch.device, dtype: torch.dtype
) -> MutableMapping[str, Any]:
    for key, value in list(values.items()):
        if isinstance(value, torch.Tensor):
            values[key] = (
                value.to(device=device, dtype=dtype)
                if key == "pixel_values"
                else value.to(device)
            )
    return values


def _longest_common_token_prefix(sequences: Sequence[torch.Tensor]) -> int:
    """Return the common token-prefix length of one-dimensional tensors."""
    if not sequences:
        return 0
    limit = min(int(sequence.numel()) for sequence in sequences)
    first = sequences[0]
    matches = torch.ones(limit, dtype=torch.bool, device=first.device)
    for sequence in sequences[1:]:
        matches.logical_and_(sequence[:limit].eq(first[:limit]))
    mismatch = torch.nonzero(~matches, as_tuple=False)
    return limit if mismatch.numel() == 0 else int(mismatch[0].item())


def _repeat_prefix_cache(past_key_values, batch_size: int) -> DynamicCache:
    """Create an independent batch-expanded DynamicCache from a prefix cache."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    legacy_cache = (
        past_key_values.to_legacy_cache()
        if isinstance(past_key_values, Cache)
        else past_key_values
    )
    cache = DynamicCache.from_legacy_cache(legacy_cache)
    cache.batch_repeat_interleave(batch_size)
    return cache


@torch.inference_mode()
def cumulative_threshold_likelihood_optimized(
    model,
    processor,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    yes_candidate: str = "YES",
    no_candidate: str = "NO",
    temperature: float = 1.0,
    threshold_batch_size: int = 10,
    prefix_cache: bool = True,
) -> Tuple[List[float], List[Dict[str, float]]]:
    """Score cumulative thresholds with shared vision and batched text inference.

    YES and NO must each tokenize to exactly one token. Video decoding, image
    preprocessing, vision encoding, projection, and compression-mask
    construction are shared across all threshold prompts. Threshold suffixes
    are processed in batches, and their common multimodal/text prefix can be
    evaluated once and reused through the language-model KV cache.
    """
    if not conversations:
        raise ValueError("at least one threshold conversation is required")
    if temperature <= 0:
        raise ValueError("likelihood temperature must be positive")
    if threshold_batch_size <= 0:
        raise ValueError("threshold_batch_size must be positive")
    tokenizer = _tokenizer(processor)
    yes_ids = tokenizer(yes_candidate, add_special_tokens=False).input_ids
    no_ids = tokenizer(no_candidate, add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            "optimized threshold logits require one-token candidates; "
            f"{yes_candidate!r}={yes_ids}, {no_candidate!r}={no_ids}"
        )

    # Load/decode the clip and run the image preprocessor exactly once.
    loaded_conversation = processor._load_multimodal_data(conversations[0])
    images = processor._gather_multimodal_data(loaded_conversation)
    if images is None:
        raise RuntimeError("threshold conversation contains no video")
    image_inputs = processor.process_images(images, return_tensors="pt")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    image_inputs = _move_processor_inputs(image_inputs, device, dtype)

    # Build the ten text prompts against the same visual-token geometry without
    # loading or preprocessing the video again.
    text_inputs = []
    for conversation in conversations:
        current_conversation = copy.deepcopy(loaded_conversation)
        # Only the final threshold question differs. Preserve the already-loaded
        # frames, timestamps, and num_frames required by the chat template.
        source_texts = [
            content
            for message in conversation
            if isinstance(message.get("content"), (list, tuple))
            for content in message["content"]
            if isinstance(content, dict) and content.get("type") == "text"
        ]
        target_texts = [
            content
            for message in current_conversation
            if isinstance(message.get("content"), (list, tuple))
            for content in message["content"]
            if isinstance(content, dict) and content.get("type") == "text"
        ]
        if len(source_texts) != 1 or len(target_texts) != 1:
            raise RuntimeError("expected exactly one threshold text item")
        target_texts[0]["text"] = source_texts[0]["text"]
        values = processor._process_conversation_without_label(
            current_conversation,
            image_inputs,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        text_inputs.append(_move_processor_inputs(values, device, dtype))

    pixel_values = image_inputs["pixel_values"]
    grid_sizes = image_inputs["grid_sizes"]
    merge_sizes = image_inputs["merge_sizes"]
    modals = image_inputs["modals"]
    batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()

    # This is the expensive visual path in VideoLLaMA3. Compute it once.
    mm_features = model.encode_images(pixel_values, grid_sizes, merge_sizes).to(device)
    mm_features = model._get_valid_visual_tokens(
        mm_features, batched_num_patches, modals
    )
    compression_mask = model._get_compression_mask(
        pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
    )

    prepared_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for values in text_inputs:
        input_ids = values["input_ids"].reshape(-1)
        attention_mask = values.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1)
        else:
            attention_mask = torch.ones_like(input_ids)

        current_features, current_mask = model._maybe_truncate_visual_tokens(
            mm_features,
            compression_mask,
            batched_num_patches,
            modals,
            input_ids,
            None,
        )
        if model.config.use_token_compression:
            (
                current_features,
                input_ids,
                attention_mask,
                _,
                _,
            ) = model._compress_visual_tokens(
                current_mask,
                current_features,
                input_ids,
                attention_mask,
                None,
                None,
            )

        inputs_embeds = model.get_model().embed_tokens(input_ids).clone()
        image_selected = input_ids == model.config.image_token_index
        inputs_embeds[image_selected] = (
            inputs_embeds[image_selected] * 0.0 + current_features
        )
        prepared_inputs.append((input_ids, attention_mask, inputs_embeds))

    final_logits_by_threshold: List[torch.Tensor | None] = [
        None
    ] * len(prepared_inputs)

    common_prefix = _longest_common_token_prefix(
        [values[0] for values in prepared_inputs]
    )
    use_prefix_cache = (
        prefix_cache
        and common_prefix > 0
        and common_prefix < min(len(values[0]) for values in prepared_inputs)
        and all(
            torch.equal(
                values[1][:common_prefix],
                prepared_inputs[0][1][:common_prefix],
            )
            for values in prepared_inputs[1:]
        )
    )

    if use_prefix_cache:
        prefix_embeddings = prepared_inputs[0][2][:common_prefix].unsqueeze(0)
        prefix_attention = prepared_inputs[0][1][:common_prefix].unsqueeze(0)
        prefix_outputs = model.get_model()(
            inputs_embeds=prefix_embeddings,
            attention_mask=prefix_attention,
            use_cache=True,
            return_dict=True,
        )
        shared_prefix_cache = prefix_outputs.past_key_values

        # Group equal-length suffixes so that the last position always predicts
        # the answer token without introducing padding-dependent indexing.
        suffix_groups: Dict[int, List[int]] = {}
        for index, (_, _, embeddings) in enumerate(prepared_inputs):
            suffix_groups.setdefault(len(embeddings) - common_prefix, []).append(index)

        for indices in suffix_groups.values():
            for offset in range(0, len(indices), threshold_batch_size):
                chunk = indices[offset : offset + threshold_batch_size]
                suffix_embeddings = torch.stack(
                    [
                        prepared_inputs[index][2][common_prefix:]
                        for index in chunk
                    ]
                )
                full_attention = torch.stack(
                    [prepared_inputs[index][1] for index in chunk]
                )
                expanded_cache = _repeat_prefix_cache(
                    shared_prefix_cache, len(chunk)
                )
                outputs = model(
                    inputs_embeds=suffix_embeddings,
                    attention_mask=full_attention,
                    past_key_values=expanded_cache,
                    use_cache=True,
                    return_dict=True,
                    num_logits_to_keep=1,
                )
                for row, index in enumerate(chunk):
                    final_logits_by_threshold[index] = outputs.logits[row, -1].float()
    else:
        # Prefix caching can be disabled for ablation. Full equal-length prompts
        # are still batched, retaining one shared visual encoding per clip.
        length_groups: Dict[int, List[int]] = {}
        for index, (_, _, embeddings) in enumerate(prepared_inputs):
            length_groups.setdefault(len(embeddings), []).append(index)
        for indices in length_groups.values():
            for offset in range(0, len(indices), threshold_batch_size):
                chunk = indices[offset : offset + threshold_batch_size]
                inputs_embeds = torch.stack(
                    [prepared_inputs[index][2] for index in chunk]
                )
                attention_mask = torch.stack(
                    [prepared_inputs[index][1] for index in chunk]
                )
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    num_logits_to_keep=1,
                )
                for row, index in enumerate(chunk):
                    final_logits_by_threshold[index] = outputs.logits[row, -1].float()

    probabilities: List[float] = []
    diagnostics: List[Dict[str, float]] = []
    for final_logits in final_logits_by_threshold:
        if final_logits is None:
            raise RuntimeError("missing final logits for a threshold")
        final_log_probs = torch.log_softmax(final_logits, dim=-1)
        yes_logit = float(final_log_probs[yes_ids[0]].item())
        no_logit = float(final_log_probs[no_ids[0]].item())
        binary_logits = torch.tensor(
            [no_logit, yes_logit], dtype=torch.float64
        ) / temperature
        probability = float(torch.softmax(binary_logits, dim=0)[1].item())
        probabilities.append(probability)
        diagnostics.append(
            {
                "yes_loglik": yes_logit,
                "no_loglik": no_logit,
                "tail_probability": probability,
            }
        )
    return probabilities, diagnostics


def build_threshold_conversation(
    video_path: Path,
    window: VideoWindow,
    sample_fps: float,
    max_frames: int,
    prompt: str,
    threshold: float,
    precise_time: bool = False,
) -> List[Dict[str, Any]]:
    """Build one cumulative ordinal threshold question without text intermediates."""
    question = (
        f"On the severity scale defined by the system, is the anomaly severity "
        f"visible in this video segment at least {threshold:.1f}? "
        "Judge only the current video's visual evidence. Answer exactly YES or NO."
    )
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(video_path),
                        "fps": sample_fps,
                        "start_time": window.start_time,
                        "end_time": window.end_time,
                        "max_frames": max_frames,
                        "precise_time": precise_time,
                    },
                },
                {"type": "text", "text": question},
            ],
        },
    ]


def is_complete(scores: Mapping[str, Any], expected_keys: Sequence[str]) -> bool:
    return set(scores) == set(expected_keys) and all(
        isinstance(scores[key], (int, float)) for key in expected_keys
    )
