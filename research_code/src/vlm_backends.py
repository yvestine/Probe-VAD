"""Backend selection and the common cumulative-scoring interface.

The command line scorer deliberately knows nothing about model-specific
processor calls.  Both backends expose ``score_thresholds`` and ``metadata``;
this keeps the VideoLLaMA3 implementation byte-for-byte compatible while
making Qwen3-VL an isolated optional dependency.
"""

from __future__ import annotations

import logging
import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
CUMULATIVE_THRESHOLDS = tuple(index / 10.0 for index in range(1, 11))


def _checkpoint_revision(model_path: str, model=None) -> str | None:
    config = getattr(model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return str(revision)
    path = Path(model_path)
    for filename in ("model_revision.txt", ".model_revision", "revision.txt"):
        marker = path / filename
        if marker.is_file():
            value = marker.read_text().strip()
            if value:
                return value
    return os.environ.get("MODEL_REVISION")


class VideoLLaMA3Backend:
    name = "videollama3"

    def __init__(self, model, processor, model_path: str):
        self.model = model
        self.processor = processor
        self.model_path = str(model_path)

    def metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "model_path": self.model_path,
            "model_type": getattr(config, "model_type", "videollama3_qwen2"),
            "model_revision": _checkpoint_revision(self.model_path, self.model),
        }

    def score_thresholds(self, video_path: Path, window, prompt_spec, args):
        from src.video_score_utils import (
            build_threshold_conversation,
            cumulative_threshold_likelihood_optimized,
            threshold_yes_no_likelihood,
        )

        conversations = [
            build_threshold_conversation(
                video_path,
                window,
                args.sample_fps,
                args.max_frames,
                prompt_spec.system_prompt,
                threshold,
                precise_time=args.precise_time,
                question_template=prompt_spec.question_template,
            )
            for threshold in CUMULATIVE_THRESHOLDS
        ]
        if args.optimized:
            return cumulative_threshold_likelihood_optimized(
                self.model,
                self.processor,
                conversations,
                yes_candidate=args.yes_candidate,
                no_candidate=args.no_candidate,
                temperature=args.likelihood_temperature,
                threshold_batch_size=args.threshold_batch_size,
                prefix_cache=args.prefix_cache,
            )

        probabilities = []
        diagnostics = []
        for threshold, conversation in zip(CUMULATIVE_THRESHOLDS, conversations):
            probability, yes_ll, no_ll = threshold_yes_no_likelihood(
                self.model,
                self.processor,
                conversation,
                yes_candidate=args.yes_candidate,
                no_candidate=args.no_candidate,
                length_normalize=args.length_normalize,
                temperature=args.likelihood_temperature,
            )
            probabilities.append(probability)
            diagnostics.append(
                {
                    "threshold": threshold,
                    "yes_loglik": yes_ll,
                    "no_loglik": no_ll,
                    "tail_probability": probability,
                }
            )
        return probabilities, diagnostics

    def score_prompt_variants(self, video_path: Path, window, prompt_specs, args):
        """Score several threshold-question suites with one visual pass.

        The optimized cumulative scorer already accepts an arbitrary list of
        threshold conversations.  Flattening prompt variants into that list
        lets it decode/encode the window once and reuse the multimodal prefix
        KV cache for all prompt/threshold suffixes.
        """
        from src.video_score_utils import (
            build_threshold_conversation,
            cumulative_threshold_likelihood_optimized,
            threshold_yes_no_likelihood,
        )

        conversations = [
            build_threshold_conversation(
                video_path,
                window,
                args.sample_fps,
                args.max_frames,
                prompt_spec.system_prompt,
                threshold,
                precise_time=args.precise_time,
                question_template=prompt_spec.question_template,
            )
            for prompt_spec in prompt_specs
            for threshold in CUMULATIVE_THRESHOLDS
        ]
        if args.optimized:
            flat_probabilities, flat_details = cumulative_threshold_likelihood_optimized(
                self.model,
                self.processor,
                conversations,
                yes_candidate=args.yes_candidate,
                no_candidate=args.no_candidate,
                temperature=args.likelihood_temperature,
                threshold_batch_size=args.threshold_batch_size,
                prefix_cache=args.prefix_cache,
            )
        else:
            flat_probabilities = []
            flat_details = []
            for conversation in conversations:
                probability, yes_ll, no_ll = threshold_yes_no_likelihood(
                    self.model,
                    self.processor,
                    conversation,
                    yes_candidate=args.yes_candidate,
                    no_candidate=args.no_candidate,
                    length_normalize=args.length_normalize,
                    temperature=args.likelihood_temperature,
                )
                flat_probabilities.append(probability)
                flat_details.append(
                    {
                        "yes_loglik": yes_ll,
                        "no_loglik": no_ll,
                        "tail_probability": probability,
                    }
                )

        result = {}
        width = len(CUMULATIVE_THRESHOLDS)
        for index, prompt_spec in enumerate(prompt_specs):
            start = index * width
            result[prompt_spec.prompt_id] = (
                flat_probabilities[start : start + width],
                flat_details[start : start + width],
            )
        return result


def _model_type(model_path: str) -> str:
    """Read config.model_type without loading model weights."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"cannot inspect model config at {model_path!r}; specify a valid local "
            "checkpoint or use --backend explicitly"
        ) from exc
    return str(getattr(config, "model_type", ""))


def resolve_backend_name(model_path: str, requested: str = "auto") -> str:
    if requested not in {"auto", "videollama3", "qwen3_vl"}:
        raise ValueError(f"unknown backend {requested!r}")
    if requested != "auto":
        return requested
    model_type = _model_type(model_path)
    if model_type == "videollama3_qwen2":
        return "videollama3"
    if model_type == "qwen3_vl":
        return "qwen3_vl"
    raise ValueError(
        f"unsupported model_type {model_type!r}; expected "
        "videollama3_qwen2 or qwen3_vl"
    )


def _load_qwen_processor(auto_processor, model_path: str):
    """Load Qwen's processor, working around Transformers 4.57.2's local
    large-vocabulary tokenizer check.

    Transformers 4.57.2 reads ``config.json`` as a dict and then accesses
    ``_config.model_type`` while loading a local tokenizer.  Qwen3-VL model
    snapshots commonly advertise ``4.57.0.dev0`` and therefore enter that
    branch.  The temporary symlinked view changes only the metadata version;
    model weights and the user's checkpoint directory are never modified.
    """
    try:
        return auto_processor.from_pretrained(model_path, trust_remote_code=True)
    except AttributeError as exc:
        if "model_type" not in str(exc) or not Path(model_path).is_dir():
            raise
        LOGGER.warning(
            "Applying the Transformers 4.57.2 local-tokenizer compatibility "
            "workaround for %s",
            model_path,
        )
        with tempfile.TemporaryDirectory(prefix="qwen3vl-processor-") as temporary:
            temporary_path = Path(temporary)
            source_path = Path(model_path)
            for item in source_path.iterdir():
                target = temporary_path / item.name
                if item.is_dir():
                    shutil.copytree(item, target, symlinks=True)
                else:
                    os.symlink(item, target)
            config_path = temporary_path / "config.json"
            try:
                config = json.loads(config_path.read_text())
                config["transformers_version"] = "4.57.3"
                config_path.unlink()
                config_path.write_text(json.dumps(config, indent=2) + "\n")
            except (OSError, json.JSONDecodeError) as config_exc:
                raise RuntimeError(
                    f"cannot prepare Qwen processor compatibility view for {model_path}"
                ) from config_exc
            return auto_processor.from_pretrained(
                str(temporary_path), trust_remote_code=True
            )


def load_backend(
    model_path: str,
    device: str | None,
    attn_implementation: str,
    backend: str = "auto",
    video_reader_backend: str = "auto",
):
    """Load one backend and return an object with ``score_thresholds``."""
    name = resolve_backend_name(model_path, backend)
    if name == "videollama3":
        from src.video_score_utils import load_model

        model, processor = load_model(model_path, device, attn_implementation)
        return VideoLLaMA3Backend(model, processor, model_path)

    # Keep Qwen imports lazy: the legacy VideoLLaMA3 environment intentionally
    # remains on Transformers 4.46.3 and must not import Qwen-only code.
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if str(target_device).startswith("cuda") else torch.float32
    if not str(target_device).startswith("cuda") and attn_implementation == "flash_attention_2":
        LOGGER.warning("flash_attention_2 requires CUDA; using eager attention on %s", target_device)
        attn_implementation = "eager"
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": target_device,
        "torch_dtype": dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()
    processor = _load_qwen_processor(AutoProcessor, model_path)
    from src.qwen3_vl_backend import Qwen3VLBackend

    return Qwen3VLBackend(model, processor, model_path, video_reader_backend)
