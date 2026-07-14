"""Text normalization and WER/CER scoring.

Normalization pipeline (in order):
    lowercase -> strip boilerplate lines (fuzzy) -> collapse whitespace
    -> optionally strip punctuation

WER/CER are computed with jiwer (imported lazily).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List, Optional

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# --------------------------------------------------------------------------- #
# Fuzzy matching (rapidfuzz if available, else a difflib fallback)
# --------------------------------------------------------------------------- #
def _partial_ratio(a: str, b: str) -> float:
    """Return a 0-100 partial similarity between two strings."""
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(a, b))
    except Exception:  # pragma: no cover - fallback path
        from difflib import SequenceMatcher

        if not a or not b:
            return 0.0
        # Approximate partial match: best window of the longer string.
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        best = 0.0
        step = max(1, len(short) // 2)
        for i in range(0, max(1, len(long) - len(short) + 1), step):
            window = long[i : i + len(short)]
            best = max(best, SequenceMatcher(None, short, window).ratio() * 100)
        # Also consider containment.
        if short in long:
            best = 100.0
        return best


def is_boilerplate(line: str, boilerplate: List[str], threshold: float) -> bool:
    """True if ``line`` fuzzy-matches any boilerplate entry at/above threshold."""
    candidate = line.strip().lower()
    if not candidate:
        return False
    for bp in boilerplate:
        if _partial_ratio(candidate, bp.strip().lower()) >= threshold:
            return True
    return False


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize_text(
    text: str,
    boilerplate: Optional[List[str]] = None,
    strip_punctuation: bool = True,
    fuzzy_threshold: float = 85,
) -> str:
    boilerplate = boilerplate or []

    # 1. lowercase
    text = (text or "").lower()

    # 2. drop boilerplate lines (fuzzy, per line)
    kept_lines = [
        line
        for line in text.splitlines()
        if not is_boilerplate(line, boilerplate, fuzzy_threshold)
    ]
    text = "\n".join(kept_lines)

    # 3. collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 4. optionally strip punctuation, then re-collapse spacing it leaves behind
    if strip_punctuation:
        text = text.translate(_PUNCT_TABLE)
        text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class PageMetrics:
    wer: Optional[float] = None
    cer: Optional[float] = None
    ref_word_count: int = 0
    hyp_word_count: int = 0
    mean_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    ocr_word_count: int = 0  # words reported by the engine (pre-normalization)
    normalized_hyp: str = field(default="", repr=False)
    normalized_ref: str = field(default="", repr=False)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("normalized_hyp", None)
        d.pop("normalized_ref", None)
        return d


def compute_metrics(
    hypothesis: str,
    reference: Optional[str],
    boilerplate: Optional[List[str]] = None,
    strip_punctuation: bool = True,
    fuzzy_threshold: float = 85,
    mean_confidence: Optional[float] = None,
    min_confidence: Optional[float] = None,
    ocr_word_count: int = 0,
) -> PageMetrics:
    """Normalize both texts and compute WER/CER (when a reference is given)."""
    norm_hyp = normalize_text(
        hypothesis, boilerplate, strip_punctuation, fuzzy_threshold
    )

    m = PageMetrics(
        hyp_word_count=len(norm_hyp.split()),
        mean_confidence=mean_confidence,
        min_confidence=min_confidence,
        ocr_word_count=ocr_word_count,
        normalized_hyp=norm_hyp,
    )

    if reference is None:
        return m

    norm_ref = normalize_text(
        reference, boilerplate, strip_punctuation, fuzzy_threshold
    )
    m.normalized_ref = norm_ref
    m.ref_word_count = len(norm_ref.split())

    if not norm_ref.strip():
        # No reference content to score against.
        return m

    import jiwer  # lazy import

    m.wer = float(jiwer.wer(norm_ref, norm_hyp))
    m.cer = float(jiwer.cer(norm_ref, norm_hyp))
    return m
