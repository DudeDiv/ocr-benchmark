import pytest

from ocrbench.engines.base import OCREngine, OCRResult, Word


def test_abstract_engine_cannot_instantiate():
    with pytest.raises(TypeError):
        OCREngine()


def test_ocrresult_confidence_stats():
    words = [
        Word("a", confidence=0.9),
        Word("b", confidence=0.5),
        Word("c", confidence=None),
    ]
    r = OCRResult(full_text="a b c", words=words, inference_seconds=1.0)
    assert r.word_count == 3
    assert r.mean_confidence == pytest.approx(0.7)
    assert r.min_confidence == 0.5


def test_ocrresult_no_confidence():
    r = OCRResult(full_text="", words=[Word("x")], inference_seconds=0.0)
    assert r.mean_confidence is None
    assert r.min_confidence is None


def test_local_engine_network_seconds_none():
    r = OCRResult(full_text="hi", inference_seconds=0.1, network_seconds=None)
    assert r.network_seconds is None
