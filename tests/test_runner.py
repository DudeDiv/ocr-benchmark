import json

import pytest

from ocrbench import run as runner
from tests.conftest import MockEngine


def test_run_over_manifest_with_mock(temp_workspace):
    pytest.importorskip("jiwer")
    cfg, hyp, gt = temp_workspace
    engine = MockEngine(name="paddle", texts=hyp)

    rows = runner.run_over_manifest(engine, cfg, device="cpu")

    assert engine.warmed is True
    assert len(rows) == 2
    assert all(r.error is None for r in rows)

    by_page = {(r.doc, r.page): r for r in rows}
    # page 1 is a perfect match
    assert by_page[("doc001", 1)].wer == 0.0
    # page 2 has one substitution (a vs the) -> WER 0.2
    assert by_page[("doc001", 2)].wer == pytest.approx(0.2, abs=1e-6)
    # confidences propagate from the engine
    assert by_page[("doc001", 1)].mean_confidence == pytest.approx(0.9)
    # paddle engine triggers the resource sampler
    assert by_page[("doc001", 1)].peak_ram_mb is not None


def test_missing_image_produces_error_row(temp_workspace):
    cfg, hyp, gt = temp_workspace
    cfg.data["manifests"]["doc001"].append(99)  # no image for page 99
    engine = MockEngine(name="paddle", texts=hyp)

    rows = runner.run_over_manifest(engine, cfg, device="cpu")
    err = [r for r in rows if r.page == 99][0]
    assert err.error is not None
    assert "not found" in err.error


def test_aggregate_and_write(temp_workspace, tmp_path):
    pytest.importorskip("jiwer")
    cfg, hyp, gt = temp_workspace
    engine = MockEngine(name="paddle", texts=hyp)
    rows = runner.run_over_manifest(engine, cfg, device="cpu")

    agg = runner.aggregate(rows, "paddle", "cpu", cfg)
    assert agg["pages_ok"] == 2
    assert agg["pages_failed"] == 0
    assert agg["mean_wer"] == pytest.approx(0.1, abs=1e-6)  # (0 + 0.2)/2
    assert agg["cost_per_page"] == 0.0
    assert agg["pages_per_second"] is not None

    out = runner.write_results(agg, cfg.results_dir(), "paddle", "cpu")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["engine"] == "paddle"

    csv_path = runner.rebuild_combined_csv(cfg.results_dir())
    content = csv_path.read_text(encoding="utf-8")
    assert "engine,device,doc,page" in content
    assert "paddle,cpu,doc001,1" in content


def test_missing_ground_truth_is_skipped_not_crashed(temp_workspace):
    pytest.importorskip("jiwer")
    cfg, hyp, gt = temp_workspace
    # Delete the ground truth for page 2 only.
    (cfg.ground_truth_dir() / "doc001" / "page_2.txt").unlink()
    engine = MockEngine(name="paddle", texts=hyp)

    rows = runner.run_over_manifest(engine, cfg, device="cpu")
    by_page = {(r.doc, r.page): r for r in rows}

    # page 1 still scored
    assert by_page[("doc001", 1)].wer == 0.0
    assert by_page[("doc001", 1)].exact_match is True
    # page 2 has no GT: wer/cer/exact_match are null, but the run did not crash
    p2 = by_page[("doc001", 2)]
    assert p2.error is None
    assert p2.wer is None and p2.cer is None and p2.exact_match is None

    agg = runner.aggregate(rows, "paddle", "cpu", cfg)
    assert agg["pages_ok"] == 2
    assert agg["pages_with_ground_truth"] == 1  # only page 1 scored


def test_cross_engine_agreement_augmentation(temp_workspace):
    pytest.importorskip("jiwer")
    cfg, hyp, gt = temp_workspace
    results_dir = cfg.results_dir()

    # docai reads the same page text; page 2 differs by one word from paddle.
    docai_texts = dict(hyp)
    docai_texts["doc001/page_2"] = "jumps over the lazy dog"  # paddle had "jumps over the lazy dog" too

    for name, texts in (("paddle", hyp), ("docai", docai_texts)):
        eng = MockEngine(name=name, texts=texts)
        rows = runner.run_over_manifest(eng, cfg, device="cpu")
        agg = runner.aggregate(rows, name, "cpu", cfg)
        runner.write_results(agg, results_dir, name, "cpu")

    # Before augmentation the agreement is unset.
    updated = runner.augment_cross_engine_agreement(results_dir)
    assert updated is True

    paddle = json.loads((results_dir / "paddle_cpu.json").read_text("utf-8"))
    docai = json.loads((results_dir / "docai_cpu.json").read_text("utf-8"))

    # Both engines produced identical text here -> agreement WER 0 on every page.
    for data in (paddle, docai):
        vals = [p["cross_engine_agreement"] for p in data["pages"]]
        assert all(v == 0.0 for v in vals)
        assert data["mean_cross_engine_agreement"] == 0.0


def test_docai_cost_applied(temp_workspace):
    pytest.importorskip("jiwer")
    cfg, hyp, gt = temp_workspace
    engine = MockEngine(name="docai", texts=hyp)
    rows = runner.run_over_manifest(engine, cfg, device="cpu")
    agg = runner.aggregate(rows, "docai", "cpu", cfg)
    assert agg["cost_per_page"] == 0.0015
    assert agg["total_cost"] == pytest.approx(0.0015 * 2)
    # docai reports network_seconds
    assert agg["mean_network_seconds"] is not None
    # docai does NOT trigger the resource sampler
    assert all(r.peak_ram_mb is None for r in rows)
