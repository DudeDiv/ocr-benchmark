"""PaddleOCR engine wrapper (PaddleOCR 3.x API).

paddlepaddle / paddleocr are imported lazily inside :meth:`_ensure_ocr` so the
module imports cleanly on machines where PaddleOCR is not installed (real runs
happen on Colab GPU).
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

from .base import OCREngine, OCRResult, Word


class PaddleEngine(OCREngine):
    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        warmup_image: Optional[str] = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        **paddle_kwargs,
    ):
        self.lang = lang
        self.device = device
        self.warmup_image = warmup_image
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.paddle_kwargs = paddle_kwargs
        self._ocr = None
        self._warmed = False

    def _ensure_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR  # lazy import

            self._ocr = PaddleOCR(
                lang=self.lang,
                device=self.device,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                **self.paddle_kwargs,
            )
        return self._ocr

    def warmup(self) -> None:
        """Run one prediction that is deliberately excluded from timing."""
        ocr = self._ensure_ocr()
        if self.warmup_image:
            ocr.predict(self.warmup_image)
        self._warmed = True

    def process(self, image_path: str) -> OCRResult:
        ocr = self._ensure_ocr()
        if not self._warmed:
            # Warm up once on first real call if warmup() was not called
            # explicitly; the warm-up call itself is not timed.
            ocr.predict(image_path)
            self._warmed = True

        start = time.perf_counter()
        results = ocr.predict(image_path)
        elapsed = time.perf_counter() - start

        words = self._parse(results)
        full_text = "\n".join(w.text for w in words)
        return OCRResult(
            full_text=full_text,
            words=words,
            inference_seconds=elapsed,
            network_seconds=None,  # local engine
            engine=self.name,
        )

    @staticmethod
    def _parse(results) -> List[Word]:
        """Flatten a PaddleOCR 3.x ``predict()`` result into a list of Words.

        ``predict()`` returns one result per input image; each result exposes
        parallel ``rec_texts`` / ``rec_scores`` / ``rec_polys`` fields (dict-like
        or attribute-style, depending on version).
        """
        words: List[Word] = []
        if not results:
            return words
        for page in results:
            texts = _get_field(page, "rec_texts") or []
            scores = _get_field(page, "rec_scores") or []
            polys = _get_field(page, "rec_polys") or []
            for i, text in enumerate(texts):
                if not text:
                    continue
                conf = float(scores[i]) if i < len(scores) and scores[i] is not None else None
                poly = polys[i] if i < len(polys) else None
                words.append(Word(text=text, bbox=_poly_to_bbox(poly), confidence=conf))
        return words


def _get_field(result: Any, key: str):
    """Read a field from a PaddleOCR 3.x result (dict-like or attribute-style)."""
    try:
        return result[key]
    except (TypeError, KeyError, IndexError):
        pass
    return getattr(result, key, None)


def _poly_to_bbox(poly: Optional[Sequence]) -> Optional[List[Tuple[float, float]]]:
    if poly is None:
        return None
    try:
        return [(float(pt[0]), float(pt[1])) for pt in poly]
    except (TypeError, IndexError, ValueError):
        return None
