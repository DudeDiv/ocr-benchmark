"""PaddleOCR engine wrapper.

paddlepaddle / paddleocr are imported lazily inside :meth:`_ensure_ocr` so the
module imports cleanly on machines where PaddleOCR is not installed (real runs
happen on Colab GPU).
"""

from __future__ import annotations

import time
from typing import List, Optional

from .base import OCREngine, OCRResult, Word


class PaddleEngine(OCREngine):
    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        warmup_image: Optional[str] = None,
        **paddle_kwargs,
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.warmup_image = warmup_image
        self.paddle_kwargs = paddle_kwargs
        self._ocr = None
        self._warmed = False

    def _ensure_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR  # lazy import

            self._ocr = PaddleOCR(
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
                **self.paddle_kwargs,
            )
        return self._ocr

    def warmup(self) -> None:
        """Run one prediction that is deliberately excluded from timing."""
        ocr = self._ensure_ocr()
        if self.warmup_image:
            ocr.ocr(self.warmup_image, cls=True)
        self._warmed = True

    def process(self, image_path: str) -> OCRResult:
        ocr = self._ensure_ocr()
        if not self._warmed:
            # Warm up once on first real call if warmup() was not called
            # explicitly; the warm-up call itself is not timed.
            ocr.ocr(image_path, cls=True)
            self._warmed = True

        start = time.perf_counter()
        raw = ocr.ocr(image_path, cls=True)
        elapsed = time.perf_counter() - start

        words = self._parse(raw)
        full_text = "\n".join(w.text for w in words)
        return OCRResult(
            full_text=full_text,
            words=words,
            inference_seconds=elapsed,
            network_seconds=None,  # local engine
            engine=self.name,
        )

    @staticmethod
    def _parse(raw) -> List[Word]:
        """Flatten PaddleOCR output into a list of Words.

        PaddleOCR returns ``[[ [box, (text, conf)], ... ]]`` (one inner list per
        image). We only ever pass a single image.
        """
        words: List[Word] = []
        if not raw:
            return words
        page = raw[0]
        if not page:
            return words
        for line in page:
            try:
                box, (text, conf) = line
            except (ValueError, TypeError):
                continue
            words.append(Word(text=text, bbox=box, confidence=float(conf)))
        return words
