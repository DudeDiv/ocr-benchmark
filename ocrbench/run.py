"""Benchmark runner CLI.

    python -m ocrbench.run --engine {paddle|docai} --device {cpu|gpu}

Produces ``results/{engine}_{device}.json`` and rebuilds a combined per-page
CSV (``results/combined_per_page.csv``) from all result JSONs present.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from .config import Config, load_config
from .engines.base import OCREngine, OCRResult
from .metrics import compute_metrics
from .resources import ResourceSampler


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #
def build_engine(engine: str, device: str, cfg: Config) -> OCREngine:
    use_gpu = device == "gpu"
    if engine == "paddle":
        from .engines.paddle_engine import PaddleEngine

        pcfg = cfg.get("paddle", {})
        return PaddleEngine(lang=pcfg.get("lang", "en"), use_gpu=use_gpu)

    if engine == "docai":
        from .engines.docai_engine import DocAIEngine

        dcfg = cfg.get("docai", {})
        return DocAIEngine(
            project_id=dcfg["project_id"],
            processor_id=dcfg["processor_id"],
            region=dcfg.get("region", "us"),
            processor_version=dcfg.get("processor_version") or None,
            mime_type=dcfg.get("mime_type", "image/png"),
            raw_dir=str(cfg.raw_dir()),
        )

    raise ValueError(f"Unknown engine: {engine!r}")


# --------------------------------------------------------------------------- #
# Per-page execution
# --------------------------------------------------------------------------- #
@dataclass
class PageRow:
    doc: str
    page: int
    engine: str
    device: str
    image_path: str
    inference_seconds: float
    network_seconds: Optional[float]
    wer: Optional[float]
    cer: Optional[float]
    ref_word_count: int
    hyp_word_count: int
    ocr_word_count: int
    mean_confidence: Optional[float]
    min_confidence: Optional[float]
    peak_cpu_percent: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    peak_gpu_util_percent: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    error: Optional[str] = None


def _read_ground_truth(gt_dir: Path, doc: str, page: int) -> Optional[str]:
    gt_path = gt_dir / doc / f"page_{page}.txt"
    if gt_path.exists():
        return gt_path.read_text(encoding="utf-8")
    return None


def process_page(
    engine: OCREngine,
    image_path: Path,
    use_sampler: bool,
    gpu: bool,
    sampler_interval: float = 0.1,
):
    """Run one page, wrapping Paddle runs with the resource sampler."""
    resources = None
    if use_sampler:
        sampler = ResourceSampler(interval=sampler_interval, gpu=gpu)
        sampler.start()
        try:
            result = engine.process(str(image_path))
        finally:
            resources = sampler.stop()
    else:
        result = engine.process(str(image_path))
    return result, resources


def run_over_manifest(
    engine: OCREngine,
    cfg: Config,
    device: str,
    images_dir: Optional[Path] = None,
    gt_dir: Optional[Path] = None,
    warmup: bool = True,
) -> List[PageRow]:
    """Core benchmarking loop over the configured manifests.

    Engine is injected (mockable). Images are expected to already exist under
    ``images_dir`` (run :mod:`ocrbench.preprocess` first).
    """
    images_dir = Path(images_dir) if images_dir else cfg.images_dir()
    gt_dir = Path(gt_dir) if gt_dir else cfg.ground_truth_dir()

    mcfg = cfg.get("metrics", {})
    strip_punct = bool(mcfg.get("strip_punctuation", True))
    fuzzy_threshold = float(mcfg.get("fuzzy_threshold", 85))
    boilerplate = cfg.boilerplate
    sampler_interval = float(cfg.get("paddle", {}).get("sampler_interval", 0.1))

    use_sampler = engine.name == "paddle"
    gpu = device == "gpu"

    if warmup:
        try:
            engine.warmup()
        except Exception:
            pass  # warm-up is best-effort and never fatal

    rows: List[PageRow] = []
    for doc, pages in cfg.manifests.items():
        for page in pages:
            image_path = images_dir / doc / f"page_{page}.png"
            if not image_path.exists():
                rows.append(
                    _error_row(doc, page, engine.name, device, image_path,
                               "image not found")
                )
                continue

            try:
                result, resources = process_page(
                    engine, image_path, use_sampler, gpu, sampler_interval
                )
            except Exception as exc:  # keep the run going on a single failure
                rows.append(
                    _error_row(doc, page, engine.name, device, image_path,
                               f"{type(exc).__name__}: {exc}")
                )
                continue

            ref = _read_ground_truth(gt_dir, doc, page)
            m = compute_metrics(
                result.full_text,
                ref,
                boilerplate=boilerplate,
                strip_punctuation=strip_punct,
                fuzzy_threshold=fuzzy_threshold,
                mean_confidence=result.mean_confidence,
                min_confidence=result.min_confidence,
                ocr_word_count=result.word_count,
            )

            row = PageRow(
                doc=doc,
                page=page,
                engine=engine.name,
                device=device,
                image_path=str(image_path),
                inference_seconds=result.inference_seconds,
                network_seconds=result.network_seconds,
                wer=m.wer,
                cer=m.cer,
                ref_word_count=m.ref_word_count,
                hyp_word_count=m.hyp_word_count,
                ocr_word_count=m.ocr_word_count,
                mean_confidence=m.mean_confidence,
                min_confidence=m.min_confidence,
            )
            if resources is not None:
                rd = resources.as_dict()
                row.peak_cpu_percent = rd["peak_cpu_percent"]
                row.peak_ram_mb = rd["peak_ram_mb"]
                row.peak_gpu_util_percent = rd["peak_gpu_util_percent"]
                row.peak_vram_mb = rd["peak_vram_mb"]
            rows.append(row)
    return rows


def _error_row(doc, page, engine, device, image_path, msg) -> PageRow:
    return PageRow(
        doc=doc, page=page, engine=engine, device=device,
        image_path=str(image_path), inference_seconds=0.0, network_seconds=None,
        wer=None, cer=None, ref_word_count=0, hyp_word_count=0, ocr_word_count=0,
        mean_confidence=None, min_confidence=None, error=msg,
    )


# --------------------------------------------------------------------------- #
# Aggregation & output
# --------------------------------------------------------------------------- #
def _avg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(mean(vals)) if vals else None


def aggregate(rows: List[PageRow], engine: str, device: str, cfg: Config) -> dict:
    ok = [r for r in rows if r.error is None]
    total_inference = sum(r.inference_seconds for r in ok)
    n_pages = len(ok)
    cost_per_page = cfg.cost_per_page(engine)

    return {
        "engine": engine,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "pages_total": len(rows),
        "pages_ok": n_pages,
        "pages_failed": len(rows) - n_pages,
        "mean_wer": _avg([r.wer for r in ok]),
        "mean_cer": _avg([r.cer for r in ok]),
        "mean_inference_seconds": _avg([r.inference_seconds for r in ok]),
        "total_inference_seconds": total_inference,
        "mean_network_seconds": _avg([r.network_seconds for r in ok]),
        "pages_per_second": (n_pages / total_inference) if total_inference else None,
        "cost_per_page": cost_per_page,
        "total_cost": cost_per_page * n_pages,
        "mean_confidence": _avg([r.mean_confidence for r in ok]),
        "peak_cpu_percent": _avg([r.peak_cpu_percent for r in ok]),
        "peak_ram_mb": _avg([r.peak_ram_mb for r in ok]),
        "peak_gpu_util_percent": _avg([r.peak_gpu_util_percent for r in ok]),
        "peak_vram_mb": _avg([r.peak_vram_mb for r in ok]),
        "pages": [asdict(r) for r in rows],
    }


def write_results(result: dict, results_dir: Path, engine: str, device: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{engine}_{device}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return out_path


CSV_FIELDS = [
    "engine", "device", "doc", "page", "inference_seconds", "network_seconds",
    "wer", "cer", "ref_word_count", "hyp_word_count", "ocr_word_count",
    "mean_confidence", "min_confidence", "peak_cpu_percent", "peak_ram_mb",
    "peak_gpu_util_percent", "peak_vram_mb", "error",
]


def rebuild_combined_csv(results_dir: Path) -> Path:
    """Merge every ``*_*.json`` result file into one per-page CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "combined_per_page.csv"

    rows: List[dict] = []
    for jf in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "pages" not in data:
            continue
        for page in data["pages"]:
            rows.append({k: page.get(k) for k in CSV_FIELDS})

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocrbench.run",
        description="Benchmark an OCR engine over the configured manifests.",
    )
    parser.add_argument("--engine", required=True, choices=["paddle", "docai"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip the engine warm-up call"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    engine = build_engine(args.engine, args.device, cfg)

    rows = run_over_manifest(
        engine, cfg, args.device, warmup=not args.no_warmup
    )
    result = aggregate(rows, args.engine, args.device, cfg)

    out_path = write_results(result, cfg.results_dir(), args.engine, args.device)
    csv_path = rebuild_combined_csv(cfg.results_dir())

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(
        f"pages_ok={result['pages_ok']}/{result['pages_total']} "
        f"mean_wer={_fmt(result['mean_wer'])} "
        f"mean_cer={_fmt(result['mean_cer'])} "
        f"mean_infer_s={_fmt(result['mean_inference_seconds'])}"
    )
    return 0


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
