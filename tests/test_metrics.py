import pytest

from ocrbench.metrics import (
    compute_metrics,
    is_boilerplate,
    normalize_text,
)

BOILERPLATE = [
    "VAJIRAM & RAVI",
    "Don't write anything in this part",
    "Introduction",
    "Marks:",
    "UPSE CSE 2025",
]


def test_lowercase_and_collapse_whitespace():
    out = normalize_text("Hello    WORLD\n\n  Foo", boilerplate=[],
                         strip_punctuation=False)
    assert out == "hello world foo"


def test_strip_punctuation():
    out = normalize_text("Hello, world! It's fine.", boilerplate=[],
                         strip_punctuation=True)
    assert out == "hello world its fine"


def test_boilerplate_exact_line_dropped():
    text = "Introduction\nThe real answer begins here\nMarks: 10"
    out = normalize_text(text, boilerplate=BOILERPLATE, strip_punctuation=False)
    assert "introduction" not in out
    assert "marks" not in out
    assert "the real answer begins here" in out


def test_boilerplate_fuzzy_match():
    # Slight OCR corruption should still be caught by fuzzy matching.
    assert is_boilerplate("vajlram & ravl", BOILERPLATE, threshold=80)
    assert is_boilerplate("UPSE  CSE 2025", BOILERPLATE, threshold=85)
    assert not is_boilerplate("a completely different line", BOILERPLATE, 85)


def test_compute_metrics_perfect_match():
    jiwer = pytest.importorskip("jiwer")
    m = compute_metrics("the quick brown fox", "the quick brown fox",
                        boilerplate=[], strip_punctuation=True)
    assert m.wer == 0.0
    assert m.cer == 0.0
    assert m.ref_word_count == 4
    assert m.hyp_word_count == 4


def test_compute_metrics_one_substitution():
    pytest.importorskip("jiwer")
    m = compute_metrics("jumps over the lazy dog", "jumps over a lazy dog",
                        boilerplate=[], strip_punctuation=True)
    # one of five words wrong
    assert m.wer == pytest.approx(0.2, abs=1e-6)
    assert m.cer is not None and m.cer > 0


def test_compute_metrics_no_reference():
    m = compute_metrics("anything", None, boilerplate=[])
    assert m.wer is None and m.cer is None
    assert m.hyp_word_count == 1


def test_boilerplate_removed_before_scoring():
    jiwer = pytest.importorskip("jiwer")
    hyp = "Introduction\nthe answer is 42"
    ref = "the answer is 42"
    m = compute_metrics(hyp, ref, boilerplate=BOILERPLATE, strip_punctuation=True)
    assert m.wer == 0.0
