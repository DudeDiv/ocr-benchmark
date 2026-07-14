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
    exact_match: Optional[bool] = None  # None when no ground-truth file exists
    ref_word_count: int = 0
    hyp_word_count: int = 0
    mean_confidence: Optional[float] = None
    min_confidence: Optional[float] = None
    ocr_word_count: int = 0  # words reported by the engine (pre-normalization)
    # Reference-free proxy (computed for every page, GT or not):
    dictionary_validity: Optional[float] = None
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

    # Reference-free: computed for every page regardless of ground truth.
    m.dictionary_validity = dictionary_validity_score(norm_hyp)

    # No ground-truth file for this page -> wer/cer/exact_match stay None and we
    # keep going. This is the accuracy subset vs. full-set distinction: only the
    # pages with a page_N.txt get a real WER/CER.
    if reference is None:
        return m

    norm_ref = normalize_text(
        reference, boilerplate, strip_punctuation, fuzzy_threshold
    )
    m.normalized_ref = norm_ref
    m.ref_word_count = len(norm_ref.split())
    m.exact_match = norm_hyp.strip() == norm_ref.strip()

    if not norm_ref.strip():
        # Reference exists but is blank (e.g. a genuinely empty page); exact_match
        # is still meaningful, but WER/CER against an empty reference is not.
        return m

    import jiwer  # lazy import

    m.wer = float(jiwer.wer(norm_ref, norm_hyp))
    m.cer = float(jiwer.cer(norm_ref, norm_hyp))
    return m


# --------------------------------------------------------------------------- #
# Reference-free proxies
#
# These do NOT establish accuracy on their own. They are supporting signals that
# extend a pattern across the full page set when only a subset has verified
# ground truth. Each has a known blind spot, documented on the function.
# --------------------------------------------------------------------------- #
_SPELL = None  # cached SpellChecker; the English wordlist load isn't free.


def _get_spell():
    """Return a cached English SpellChecker, or None if pyspellchecker is absent."""
    global _SPELL
    if _SPELL is None:
        try:
            from spellchecker import SpellChecker

            _SPELL = SpellChecker(language="en")
        except Exception:  # pragma: no cover - optional dependency
            _SPELL = False  # sentinel: tried and unavailable
    return _SPELL or None


def dictionary_validity_score(text: str) -> Optional[float]:
    """Fraction of alphabetic output words found in an English wordlist.

    A high value means the OCR is emitting real words; a low value flags garbage.

    Blind spot: it cannot catch valid-word substitutions. "cat" mis-read as "cot"
    scores as perfectly valid, because both are real words. So this detects
    gibberish output, not wrong-but-plausible output. Returns None if
    pyspellchecker is unavailable or there are no alphabetic words to judge.
    """
    spell = _get_spell()
    if spell is None:
        return None
    words = [w for w in (text or "").split() if w.isalpha()]
    if not words:
        return None
    unknown = spell.unknown(words)
    return (len(words) - len(unknown)) / len(words)


def cross_engine_agreement(
    text_a: str,
    text_b: str,
    boilerplate: Optional[List[str]] = None,
    strip_punctuation: bool = True,
    fuzzy_threshold: float = 85,
    prenormalized: bool = False,
) -> Optional[float]:
    """WER between two engines' outputs for the same page (``text_a`` as reference).

    Low means the engines agree; high means they diverge. Pass already-normalized
    text with ``prenormalized=True`` to skip re-normalization.

    Blind spot: agreement is not correctness. Where both engines make the *same*
    mistake this reads as perfect agreement, and a high value tells you the
    engines disagree, not which one is right. It localizes divergence for review;
    it does not adjudicate it. Returns None if the reference side is empty or
    jiwer is unavailable.
    """
    if not prenormalized:
        text_a = normalize_text(text_a, boilerplate, strip_punctuation, fuzzy_threshold)
        text_b = normalize_text(text_b, boilerplate, strip_punctuation, fuzzy_threshold)

    if not text_a.strip():
        return None  # WER is undefined against an empty reference

    try:
        import jiwer  # lazy import
    except Exception:  # pragma: no cover - optional dependency
        return None
    return float(jiwer.wer(text_a, text_b))
