"""Abstract OCR engine interface and result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean
from typing import List, Optional, Sequence, Tuple

# A bounding box as four (x, y) corner points in image pixel coordinates.
BBox = Sequence[Tuple[float, float]]


@dataclass
class Word:
    """A single recognized token with geometry and confidence."""

    text: str
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None


@dataclass
class OCRResult:
    """Normalized output returned by every engine.

    Attributes
    ----------
    full_text:
        The complete recognized text for the page (newline-joined lines).
    words:
        Per-token results (text, bbox, confidence).
    inference_seconds:
        Wall-clock time spent doing OCR (model/API compute).
    network_seconds:
        Time attributable to network round-trips. ``None`` for local engines.
    engine:
        Name of the engine that produced this result.
    """

    full_text: str
    words: List[Word] = field(default_factory=list)
    inference_seconds: float = 0.0
    network_seconds: Optional[float] = None
    engine: str = ""

    @property
    def word_count(self) -> int:
        return len(self.words)

    def confidences(self) -> List[float]:
        return [w.confidence for w in self.words if w.confidence is not None]

    @property
    def mean_confidence(self) -> Optional[float]:
        vals = self.confidences()
        return float(mean(vals)) if vals else None

    @property
    def min_confidence(self) -> Optional[float]:
        vals = self.confidences()
        return float(min(vals)) if vals else None


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    name: str = "base"

    @abstractmethod
    def process(self, image_path: str) -> OCRResult:
        """Run OCR on a single image and return an :class:`OCRResult`."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional one-time warm-up; excluded from timed runs. No-op by default."""
        return None
