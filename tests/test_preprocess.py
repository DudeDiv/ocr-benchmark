import pytest

from ocrbench.preprocess import _pdf_path_for


def test_exact_match(tmp_path):
    (tmp_path / "GS-1.pdf").write_bytes(b"%PDF-1.4")
    found = _pdf_path_for("GS-1", tmp_path)
    assert found.name == "GS-1.pdf"


def test_case_insensitive_match(tmp_path):
    (tmp_path / "gs-1.pdf").write_bytes(b"%PDF-1.4")
    found = _pdf_path_for("GS-1", tmp_path)
    assert found.name == "gs-1.pdf"


def test_suffix_wildcard_match(tmp_path):
    (tmp_path / "GS-1_scanned_copy.PDF").write_bytes(b"%PDF-1.4")
    found = _pdf_path_for("GS-1", tmp_path)
    assert found.name == "GS-1_scanned_copy.PDF"


def test_multiple_matches_pick_deterministically(tmp_path):
    # Two files both satisfy the "GS-1*.pdf" pattern; the choice must be stable
    # across runs rather than depending on filesystem iteration order.
    (tmp_path / "GS-1_v2.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "GS-1_v1.pdf").write_bytes(b"%PDF-1.4")
    first = _pdf_path_for("GS-1", tmp_path)
    second = _pdf_path_for("GS-1", tmp_path)
    assert first.name == second.name == "GS-1_v1.pdf"  # alphabetically first


def test_missing_pdf_error_lists_doc_dir_and_files(tmp_path):
    (tmp_path / "GS-2.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("irrelevant")

    with pytest.raises(FileNotFoundError) as exc_info:
        _pdf_path_for("GS-1", tmp_path)

    msg = str(exc_info.value)
    assert "GS-1" in msg
    assert str(tmp_path) in msg
    assert "GS-2.pdf" in msg  # what was actually found is listed


def test_missing_directory_reports_none_found(tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError) as exc_info:
        _pdf_path_for("GS-1", missing_dir)
    assert "GS-1" in str(exc_info.value)
