"""Versioned prompt definitions for cumulative ordinal video scoring.

The sensitivity suite deliberately changes only the threshold question.  The
system prompt, ordinal scale, candidates and inference parameters stay fixed
so that a run can be interpreted as a prompt-only ablation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Mapping

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful video anomaly detector for public surveillance scenes. "
    "Judge only visible evidence in the current video segment. Use an ordered "
    "anomaly-severity scale from 0 to 1: 0 means ordinary safe activity; 0.5 "
    "means clearly concerning or plausibly anomalous activity; and 1 means "
    "unmistakably severe, dangerous, violent, criminal, or emergency activity. "
    "Intermediate thresholds preserve this order. Do not infer events that are "
    "not visually supported."
)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    system_prompt: str
    question_template: str

    def question(self, threshold: float) -> str:
        try:
            parsed = list(Formatter().parse(self.question_template))
            placeholders = [(field, spec, conversion) for _, field, spec, conversion in parsed if field is not None]
        except ValueError as exc:
            raise ValueError(f"invalid threshold template for prompt {self.prompt_id!r}: {exc}") from exc
        if (
            len(placeholders) != 1
            or placeholders[0][0] != "threshold"
            or placeholders[0][1] not in ("", ".1f")
            or placeholders[0][2] is not None
        ):
            raise ValueError(
                f"prompt {self.prompt_id!r} must contain exactly one "
                "legal {threshold:.1f} placeholder"
            )
        try:
            value = self.question_template.format(threshold=threshold)
        except (KeyError, ValueError, IndexError, AttributeError) as exc:
            raise ValueError(
                f"invalid threshold template for prompt {self.prompt_id!r}: {exc}"
            ) from exc
        if "{threshold" in value:
            raise ValueError(f"unresolved threshold placeholder in {self.prompt_id!r}")
        return value

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


PROMPT_SPECS: dict[str, PromptSpec] = {
    "baseline": PromptSpec(
        "baseline",
        DEFAULT_SYSTEM_PROMPT,
        "On the severity scale defined by the system, is the anomaly severity "
        "visible in this video segment at least {threshold:.1f}? Judge only the "
        "current video's visual evidence. Answer exactly YES or NO.",
    ),
    "visible_evidence": PromptSpec(
        "visible_evidence",
        DEFAULT_SYSTEM_PROMPT,
        "Considering only what is visibly shown in this video segment, is its "
        "anomaly severity at least {threshold:.1f} on the defined scale? Answer "
        "exactly YES or NO.",
    ),
    "reach_level": PromptSpec(
        "reach_level",
        DEFAULT_SYSTEM_PROMPT,
        "Does the visible activity in this video segment reach severity level "
        "{threshold:.1f} or higher on the defined anomaly scale? Answer exactly "
        "YES or NO.",
    ),
    "no_less_than": PromptSpec(
        "no_less_than",
        DEFAULT_SYSTEM_PROMPT,
        "Using only the current clip's visual evidence, is this segment no less "
        "anomalous than severity {threshold:.1f} on the defined scale? Answer "
        "exactly YES or NO.",
    ),
    "rated_above": PromptSpec(
        "rated_above",
        DEFAULT_SYSTEM_PROMPT,
        "Would the anomaly visible in this video segment be rated {threshold:.1f} "
        "or above on the defined severity scale? Answer exactly YES or NO.",
    ),
}


def get_prompt_spec(
    variant: str = "baseline",
    prompt_file: Path | None = None,
    system_override: str | None = None,
    question_override: str | None = None,
) -> PromptSpec:
    """Load one built-in or JSON prompt and apply explicit CLI overrides."""
    if prompt_file is not None:
        try:
            value: Any = json.loads(prompt_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read prompt file {prompt_file}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("prompt file must contain a JSON object")
        # Accept both a single prompt object and the versioned suite file in
        # configs/covas_prompt_suite.json.  A suite is still resolved to one
        # immutable PromptSpec so a run manifest records the exact text used.
        if "variants" in value:
            variants = value.get("variants")
            if not isinstance(variants, Mapping) or variant not in variants:
                available = ", ".join(sorted(variants)) if isinstance(variants, Mapping) else ""
                raise ValueError(
                    f"prompt suite has no variant {variant!r}; choices: {available}"
                )
            selected = variants[variant]
            if not isinstance(selected, Mapping):
                raise ValueError(f"prompt suite variant {variant!r} must be an object")
            prompt_id = str(selected.get("prompt_id", variant))
            system_prompt = str(
                selected.get("system_prompt", value.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
            )
            question_template = str(selected.get("question_template", ""))
        else:
            prompt_id = str(value.get("prompt_id", prompt_file.stem))
            system_prompt = str(value.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
            question_template = str(value.get("question_template", ""))
        spec = PromptSpec(prompt_id, system_prompt, question_template)
    else:
        try:
            spec = PROMPT_SPECS[variant]
        except KeyError as exc:
            choices = ", ".join(sorted(PROMPT_SPECS))
            raise ValueError(f"unknown prompt variant {variant!r}; choices: {choices}") from exc

    if system_override is not None:
        spec = PromptSpec(spec.prompt_id, system_override, spec.question_template)
    if question_override is not None:
        spec = PromptSpec(spec.prompt_id, spec.system_prompt, question_override)
    # Validate before model loading so malformed experiments fail cheaply.
    spec.question(0.1)
    return spec


def prompt_suite_dict() -> dict[str, dict[str, str]]:
    return {prompt_id: spec.to_dict() for prompt_id, spec in PROMPT_SPECS.items()}
