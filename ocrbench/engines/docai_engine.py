"""Google Document AI engine wrapper.

google-cloud-documentai is imported lazily so the module imports without the
dependency present. Credentials are read from the GOOGLE_APPLICATION_CREDENTIALS
environment variable (path to the service-account JSON key).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from .base import OCREngine, OCRResult, Word


class DocAIEngine(OCREngine):
    name = "docai"

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        region: str = "us",
        processor_version: Optional[str] = None,
        mime_type: str = "image/png",
        raw_dir: Optional[str] = None,
    ):
        self.project_id = project_id
        self.processor_id = processor_id
        self.region = region
        self.processor_version = processor_version
        self.mime_type = mime_type
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud import documentai

            opts = ClientOptions(
                api_endpoint=f"{self.region}-documentai.googleapis.com"
            )
            self._client = documentai.DocumentProcessorServiceClient(
                client_options=opts
            )
        return self._client

    def _processor_name(self) -> str:
        client = self._ensure_client()
        if self.processor_version:
            return client.processor_version_path(
                self.project_id,
                self.region,
                self.processor_id,
                self.processor_version,
            )
        return client.processor_path(
            self.project_id, self.region, self.processor_id
        )

    def process(self, image_path: str) -> OCRResult:
        from google.cloud import documentai

        client = self._ensure_client()
        with open(image_path, "rb") as fh:
            content = fh.read()

        raw_document = documentai.RawDocument(
            content=content, mime_type=self.mime_type
        )
        request = documentai.ProcessRequest(
            name=self._processor_name(), raw_document=raw_document
        )

        start = time.perf_counter()
        response = client.process_document(request=request)
        elapsed = time.perf_counter() - start

        document = response.document
        self._save_raw(image_path, document)

        words, full_text = self._parse(document)
        return OCRResult(
            full_text=full_text or (document.text or ""),
            words=words,
            inference_seconds=elapsed,
            # The whole call is a network round-trip; total time is dominated by
            # it, so we attribute the same measured wall-clock as network time.
            network_seconds=elapsed,
            engine=self.name,
        )

    def _save_raw(self, image_path: str, document) -> None:
        if self.raw_dir is None:
            return
        from google.cloud import documentai

        out_dir = self.raw_dir / "docai"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        # Include the immediate parent (doc id) to avoid page_1 collisions.
        doc_id = Path(image_path).parent.name
        out_path = out_dir / f"{doc_id}__{stem}.json"
        payload = documentai.Document.to_json(document)
        with out_path.open("w", encoding="utf-8") as fh:
            # to_json already returns a JSON string; re-dump for stable spacing.
            fh.write(json.dumps(json.loads(payload), ensure_ascii=False, indent=2))

    @staticmethod
    def _parse(document):
        """Extract per-token text/bbox/confidence from a Document AI response."""
        words: List[Word] = []
        text = document.text or ""
        lines_out: List[str] = []

        for page in document.pages:
            for token in page.tokens:
                seg_text = _text_from_anchor(token.layout.text_anchor, text)
                if not seg_text.strip():
                    continue
                conf = getattr(token.layout, "confidence", None)
                bbox = _bbox_from_poly(getattr(token.layout, "bounding_poly", None))
                words.append(
                    Word(
                        text=seg_text.strip(),
                        bbox=bbox,
                        confidence=float(conf) if conf is not None else None,
                    )
                )
            # Preserve line structure for full_text using the page's lines.
            for line in page.lines:
                seg = _text_from_anchor(line.layout.text_anchor, text)
                if seg.strip():
                    lines_out.append(seg.strip())

        full_text = "\n".join(lines_out) if lines_out else text
        return words, full_text


def _text_from_anchor(text_anchor, full_text: str) -> str:
    """Resolve a Document AI TextAnchor to its substring of ``full_text``."""
    if text_anchor is None or not getattr(text_anchor, "text_segments", None):
        return ""
    parts = []
    for seg in text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index else 0
        end = int(seg.end_index)
        parts.append(full_text[start:end])
    return "".join(parts)


def _bbox_from_poly(bounding_poly):
    if bounding_poly is None:
        return None
    verts = getattr(bounding_poly, "normalized_vertices", None) or getattr(
        bounding_poly, "vertices", None
    )
    if not verts:
        return None
    return [(getattr(v, "x", 0.0), getattr(v, "y", 0.0)) for v in verts]
