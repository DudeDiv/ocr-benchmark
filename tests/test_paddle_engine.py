"""Tests for the PaddleOCR 3.x result-parsing logic (no paddleocr install needed).

paddleocr is never imported here; we only exercise ``PaddleEngine._parse``
against fake result objects shaped like what ``predict()`` returns in 3.x.
"""

from ocrbench.engines.paddle_engine import PaddleEngine


class DictLikeResult(dict):
    """Mimics a PaddleOCR 3.x result that supports item access."""


class AttrLikeResult:
    """Mimics a PaddleOCR 3.x result that supports attribute access."""

    def __init__(self, rec_texts, rec_scores, rec_polys):
        self.rec_texts = rec_texts
        self.rec_scores = rec_scores
        self.rec_polys = rec_polys


def test_parse_dict_like_result():
    page = DictLikeResult(
        rec_texts=["hello", "world"],
        rec_scores=[0.95, 0.80],
        rec_polys=[[(0, 0), (1, 0), (1, 1), (0, 1)], [(2, 0), (3, 0), (3, 1), (2, 1)]],
    )
    words = PaddleEngine._parse([page])
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].confidence == 0.95
    assert words[1].bbox == [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]


def test_parse_attribute_like_result():
    page = AttrLikeResult(
        rec_texts=["answer"],
        rec_scores=[0.99],
        rec_polys=[[(0, 0), (1, 0), (1, 1), (0, 1)]],
    )
    words = PaddleEngine._parse([page])
    assert len(words) == 1
    assert words[0].text == "answer"
    assert words[0].confidence == 0.99


def test_parse_empty_results():
    assert PaddleEngine._parse([]) == []
    assert PaddleEngine._parse(None) == []


def test_parse_skips_empty_text():
    page = DictLikeResult(rec_texts=["", "real"], rec_scores=[0.5, 0.9], rec_polys=[None, None])
    words = PaddleEngine._parse([page])
    assert [w.text for w in words] == ["real"]


def test_parse_missing_polys_field():
    # Some result variants might omit rec_polys entirely; must not crash.
    page = DictLikeResult(rec_texts=["ok"], rec_scores=[0.7])
    words = PaddleEngine._parse([page])
    assert words[0].text == "ok"
    assert words[0].bbox is None


def test_engine_constructor_uses_device_not_use_gpu():
    # This is a regression check for the PaddleOCR 3.x migration: the engine
    # must accept `device`, not the old boolean `use_gpu`.
    engine = PaddleEngine(lang="en", device="gpu")
    assert engine.device == "gpu"
    assert not hasattr(engine, "use_gpu")
