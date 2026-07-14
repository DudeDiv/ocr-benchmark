"""Weighted scorecard across accuracy, latency, cost, throughput, resources.

Each dimension maps to a metric in the per-engine aggregate JSON. Metrics are
normalized to 0..1 across the engines being compared (1 = best), then combined
with the weights from ``config.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, load_config

# dimension -> (metric key in aggregate, direction) where "lower"/"higher" says
# which raw value is better.
DIMENSIONS = {
    "accuracy": ("mean_wer", "lower"),      # lower word error rate is better
    "latency": ("mean_inference_seconds", "lower"),
    "cost": ("cost_per_page", "lower"),
    "throughput": ("pages_per_second", "higher"),
    # resources is a composite of the four peaks (see _resource_index)
    "resources": ("_resource_index", "lower"),
}

_RESOURCE_KEYS = [
    "peak_cpu_percent",
    "peak_ram_mb",
    "peak_gpu_util_percent",
    "peak_vram_mb",
]


def _resource_index(agg: dict) -> Optional[float]:
    """Mean of the available resource peaks; None if the engine reported none.

    Engines that do no local work (e.g. Doc AI) report no peaks and get None,
    which :func:`_normalize` treats as the best (least local burden) score.
    """
    vals = [agg.get(k) for k in _RESOURCE_KEYS]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _metric_value(agg: dict, key: str) -> Optional[float]:
    if key == "_resource_index":
        return _resource_index(agg)
    v = agg.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _normalize(values: List[Optional[float]], direction: str) -> List[float]:
    """Min-max normalize to 0..1 with 1 = best, per the direction.

    None values (metric unavailable) are scored 1.0 for "lower is better"
    dimensions (no measured burden) and 0.0 for "higher is better".
    """
    present = [v for v in values if v is not None]
    default = 1.0 if direction == "lower" else 0.0
    if not present:
        return [default for _ in values]

    lo, hi = min(present), max(present)
    span = hi - lo
    out: List[float] = []
    for v in values:
        if v is None:
            out.append(default)
        elif span == 0:
            out.append(1.0)  # all equal -> tie at best
        elif direction == "lower":
            out.append((hi - v) / span)
        else:  # higher is better
            out.append((v - lo) / span)
    return out


def build_scorecard(
    aggregates: List[dict], weights: Dict[str, float]
) -> dict:
    """Compute weighted scores for a list of per-engine aggregate dicts."""
    labels = [f"{a.get('engine')}_{a.get('device')}" for a in aggregates]

    # Normalize each dimension across engines.
    normalized: Dict[str, List[float]] = {}
    raw: Dict[str, List[Optional[float]]] = {}
    for dim, (key, direction) in DIMENSIONS.items():
        raw_vals = [_metric_value(a, key) for a in aggregates]
        raw[dim] = raw_vals
        normalized[dim] = _normalize(raw_vals, direction)

    total_weight = sum(weights.get(d, 0.0) for d in DIMENSIONS) or 1.0

    engines_out = []
    for i, label in enumerate(labels):
        dim_scores = {}
        weighted = 0.0
        for dim in DIMENSIONS:
            score = normalized[dim][i]
            w = weights.get(dim, 0.0)
            dim_scores[dim] = {
                "raw": raw[dim][i],
                "normalized": round(score, 4),
                "weight": w,
                "weighted": round(score * w, 4),
            }
            weighted += score * w
        engines_out.append(
            {
                "label": label,
                "engine": aggregates[i].get("engine"),
                "device": aggregates[i].get("device"),
                "total_score": round(weighted / total_weight, 4),
                "dimensions": dim_scores,
            }
        )

    engines_out.sort(key=lambda e: e["total_score"], reverse=True)
    return {
        "weights": weights,
        "dimensions": {d: DIMENSIONS[d] for d in DIMENSIONS},
        "engines": engines_out,
        "winner": engines_out[0]["label"] if engines_out else None,
    }


def score_from_results_dir(
    results_dir: Path, weights: Dict[str, float]
) -> dict:
    aggregates = []
    for jf in sorted(Path(results_dir).glob("*.json")):
        if jf.name in ("scorecard.json",):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "engine" in data and "pages" in data:
            aggregates.append(data)
    return build_scorecard(aggregates, weights)


def _format_table(scorecard: dict) -> str:
    lines = ["Scorecard (1.0 = best):", ""]
    header = f"{'engine':<16}{'total':>8}  " + "  ".join(
        f"{d:>11}" for d in DIMENSIONS
    )
    lines.append(header)
    lines.append("-" * len(header))
    for e in scorecard["engines"]:
        row = f"{e['label']:<16}{e['total_score']:>8.3f}  " + "  ".join(
            f"{e['dimensions'][d]['normalized']:>11.3f}" for d in DIMENSIONS
        )
        lines.append(row)
    lines.append("")
    lines.append(f"Winner: {scorecard['winner']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m ocrbench.scorecard",
        description="Build a weighted scorecard from results/*.json.",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)

    cfg: Config = load_config(args.config)
    scorecard = score_from_results_dir(cfg.results_dir(), cfg.weights)

    out_path = cfg.results_dir() / "scorecard.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, ensure_ascii=False, indent=2)

    print(_format_table(scorecard))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
