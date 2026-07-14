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
