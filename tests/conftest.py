"""Shared test fixtures: a mocked OCR engine and a temp config."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocrbench.config import Config
from ocrbench.engines.base import OCREngine, OCRResult, Word


class MockEngine(OCREngine):
    """Deterministic engine driven by a {(doc, page) or image-stem -> text} map."""

    def __init__(self, name="paddle", texts=None, per_word_conf=0.9, infer_s=0.05,
                 raise_error=None):
        self.name = name
        self.texts = texts or {}
        self.per_word_conf = per_word_conf
        self.infer_s = infer_s
        self.raise_error = raise_error  # exception instance/class to raise on process()
        self.warmed = False
        self.calls = []

    def warmup(self) -> None:
        self.warmed = True

    def process(self, image_path: str) -> OCRResult:
        self.calls.append(image_path)
        if self.raise_error is not None:
            raise self.raise_error
        p = Path(image_path)
        key = f"{p.parent.name}/{p.stem}"
        text = self.texts.get(key, "hello world")
        words = [
            Word(text=tok, bbox=[(0, 0), (1, 0), (1, 1), (0, 1)],
                 confidence=self.per_word_conf)
            for tok in text.split()
        ]
        net = None if self.name == "paddle" else self.infer_s
        return OCRResult(
            full_text=text,
            words=words,
            inference_seconds=self.infer_s,
            network_seconds=net,
            engine=self.name,
        )


@pytest.fixture
def mock_engine():
    return MockEngine()


@pytest.fixture
def temp_workspace(tmp_path):
    """Build images + ground truth for a two-page manifest and return a Config."""
    images_dir = tmp_path / "work" / "images"
    gt_dir = tmp_path / "ground_truth"
    results_dir = tmp_path / "results"

    manifests = {"doc001": [1, 2]}
    hyp = {
        "doc001/page_1": "the quick brown fox",
        "doc001/page_2": "jumps over the lazy dog",
    }
    gt = {
        "doc001/page_1": "the quick brown fox",
        "doc001/page_2": "jumps over a lazy dog",  # one substitution -> WER > 0
    }

    for key, text in hyp.items():
        doc, page = key.split("/")
        img = images_dir / doc / f"{page}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"\x89PNG\r\n")  # placeholder; MockEngine ignores content
    for key, text in gt.items():
        doc, page = key.split("/")
        txt = gt_dir / doc / f"{page}.txt"
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(text, encoding="utf-8")

    data = {
        "paths": {
            "input_pdfs": "pdfs",
            "work_dir": "work",
            "images_dir": str(images_dir),
            "raw_dir": str(tmp_path / "work" / "raw"),
            "ground_truth_dir": str(gt_dir),
            "results_dir": str(results_dir),
        },
        "render": {"dpi": 300},
        "paddle": {
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "sampler_interval": 0.01,
        },
        "manifests": manifests,
        "costs": {"paddle_per_page": 0.0, "docai_per_page": 0.0015},
        "metrics": {"strip_punctuation": True, "fuzzy_threshold": 85},
        "boilerplate": ["Introduction", "Marks:"],
        "scorecard": {
            "weights": {
                "accuracy": 0.4, "latency": 0.2, "cost": 0.15,
                "throughput": 0.15, "resources": 0.1,
            }
        },
    }
    return Config(data, base_dir=tmp_path), hyp, gt
