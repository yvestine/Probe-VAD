"""Experiment provenance and resume guards for root-level COVAS runs."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)
MANIFEST_NAME = "_run_manifest.json"
MANIFEST_SCHEMA = "1.0"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def build_manifest(args, prompt_spec, backend_name: str, model_metadata: Mapping[str, Any]) -> dict[str, Any]:
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        runtime["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers

        runtime["transformers"] = transformers.__version__
    except Exception:
        pass
    values = {
        "schema_version": MANIFEST_SCHEMA,
        "backend": backend_name,
        "model": dict(model_metadata),
        "prompt": prompt_spec.to_dict(),
        "candidates": {
            "yes": args.yes_candidate,
            "no": args.no_candidate,
        },
        "scoring": {
            "thresholds": [round(index / 10.0, 1) for index in range(1, 11)],
            "frame_interval": args.frame_interval,
            "window_seconds": args.window_seconds,
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
            "precise_time": args.precise_time,
            "likelihood_temperature": args.likelihood_temperature,
            "length_normalize": args.length_normalize,
            "optimized": args.optimized,
            "threshold_batch_size": args.threshold_batch_size,
            "prefix_cache": args.prefix_cache,
            "monotonic_projection": args.monotonic_projection,
            "video_reader_backend": args.video_reader_backend,
        },
        "sampling": {
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
            "temperature": args.likelihood_temperature,
            "do_sample": False,
        },
        "runtime": runtime,
    }
    canonical = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    values["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return values


def ensure_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = output_dir / MANIFEST_NAME
    if path.is_file():
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot parse existing run manifest {path}: {exc}") from exc
        if current.get("config_sha256") != manifest.get("config_sha256"):
            raise RuntimeError(
                f"run manifest mismatch in {output_dir}; use a new OUTPUT_DIR "
                "for a different model, prompt, or scoring configuration"
            )
        return

    score_files = [
        item
        for item in output_dir.glob("*.json")
        if item.name != MANIFEST_NAME
    ]
    if score_files:
        LOGGER.warning(
            "legacy score directory %s has no manifest; provenance cannot be "
            "validated for existing files",
            output_dir,
        )
    _atomic_write_json(path, dict(manifest))
