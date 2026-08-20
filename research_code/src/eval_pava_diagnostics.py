"""Offline PAVA diagnostics for saved ten-threshold clip details.

The scorer stores one detail dictionary per clip/window under
``_threshold_details``.  This utility never loads a model and reports the
five statistics used by the PAVA appendix ablation:

* fraction of clips with at least one monotonicity violation;
* mean number of violations per clip;
* mean absolute raw-to-PAVA probability correction;
* maximum absolute probability correction;
* maximum absolute change in the scalar mean score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(details_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(details_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot parse {path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        for clip_key, detail in payload.items():
            if str(clip_key).startswith("_") or not isinstance(detail, dict):
                continue
            thresholds = detail.get("thresholds")
            adjusted = detail.get("monotonic_tail_probabilities")
            if not isinstance(thresholds, list) or len(thresholds) != 10:
                raise ValueError(f"{path}:{clip_key} lacks ten raw thresholds")
            raw = np.asarray(
                [float(item["tail_probability"]) for item in thresholds],
                dtype=np.float64,
            )
            pava = np.asarray(adjusted, dtype=np.float64)
            if pava.shape != (10,):
                raise ValueError(f"{path}:{clip_key} lacks ten PAVA probabilities")
            violations = int(
                detail.get(
                    "monotonic_violations",
                    sum(raw[index] < raw[index + 1] for index in range(9)),
                )
            )
            raw_score = float(np.mean(raw))
            pava_score = float(detail.get("score", np.mean(pava)))
            rows.append(
                {
                    "file": path.name,
                    "clip": str(clip_key),
                    "violations": violations,
                    "raw_score": raw_score,
                    "pava_score": pava_score,
                    "abs_delta": np.abs(raw - pava),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("no threshold-detail clips found")
    violations = np.asarray([row["violations"] for row in rows], dtype=np.float64)
    corrections = np.concatenate([row["abs_delta"] for row in rows])
    scalar_delta = np.asarray(
        [abs(row["raw_score"] - row["pava_score"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "clips": len(rows),
        "clips_with_violation": int(np.count_nonzero(violations > 0)),
        "violation_clip_ratio": float(np.mean(violations > 0)),
        "mean_violations_per_clip": float(np.mean(violations)),
        "mean_absolute_probability_correction": float(np.mean(corrections)),
        "max_absolute_probability_correction": float(np.max(corrections)),
        "max_absolute_scalar_score_difference": float(np.max(scalar_delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = _load_rows(args.details_dir)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pava_diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / "pava_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    lines = [
        "| Statistic | Value |",
        "|---|---:|",
        f"| Clips | {summary['clips']} |",
        f"| Violation clip ratio | {summary['violation_clip_ratio']:.8f} |",
        f"| Mean violations per clip | {summary['mean_violations_per_clip']:.8f} |",
        f"| Mean absolute probability correction | {summary['mean_absolute_probability_correction']:.8f} |",
        f"| Max absolute probability correction | {summary['max_absolute_probability_correction']:.8f} |",
        f"| Max absolute scalar score difference | {summary['max_absolute_scalar_score_difference']:.8e} |",
        "",
    ]
    (args.output_dir / "pava_diagnostics.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
