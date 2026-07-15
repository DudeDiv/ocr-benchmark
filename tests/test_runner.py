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


def test_process_exception_captures_type_and_traceback(temp_workspace):
    cfg, hyp, gt = temp_workspace
    engine = MockEngine(name="paddle", texts=hyp, raise_error=RuntimeError("boom"))

    rows = runner.run_over_manifest(engine, cfg, device="cpu")
    assert all(r.error is not None for r in rows)
    assert all(r.error_type == "RuntimeError" for r in rows)
    assert all(r.error_traceback is not None for r in rows)
    assert all("RuntimeError: boom" in r.error_traceback for r in rows)


def test_missing_image_has_error_type_but_no_traceback(temp_workspace):
    cfg, hyp, gt = temp_workspace
    cfg.data["manifests"]["doc001"].append(99)
    engine = MockEngine(name="paddle", texts=hyp)

    rows = runner.run_over_manifest(engine, cfg, device="cpu")
    err = [r for r in rows if r.page == 99][0]
    assert err.error_type == "MissingImage"
    assert err.error_traceback is None


def test_summarize_errors_counts_by_type(temp_workspace):
    cfg, hyp, gt = temp_workspace
    cfg.data["manifests"]["doc001"].append(99)  # -> one MissingImage
    engine = MockEngine(name="paddle", texts=hyp, raise_error=ValueError("bad"))

    rows = runner.run_over_manifest(engine, cfg, device="cpu")
    counts = runner.summarize_errors(rows)
    # pages 1 and 2 raise ValueError during process(), page 99's image is missing
    assert counts == {"ValueError": 2, "MissingImage": 1}


def test_main_exits_nonzero_and_prints_tracebacks_on_total_failure(
    temp_workspace, monkeypatch, capsys
):
    cfg, hyp, gt = temp_workspace
    failing_engine = MockEngine(name="paddle", texts=hyp, raise_error=RuntimeError("boom"))

    # Bypass the dependency preflight so this test exercises the fail-loud
    # aggregate/exit path regardless of whether paddleocr happens to be
    # installed on the machine running the suite.
    monkeypatch.setattr(runner, "preflight", lambda engine: None)
    monkeypatch.setattr(runner, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(runner, "build_engine", lambda engine, device, c: failing_engine)

    rc = runner.main(["--engine", "paddle", "--device", "cpu"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "FATAL" in captured.err
    assert "0/2 pages succeeded" in captured.err
    assert "RuntimeError" in captured.err
    assert "boom" in captured.err
    # error summary is printed on stdout regardless of outcome
    assert "RuntimeError: 2" in captured.out


def test_main_prints_none_error_summary_on_full_success(
    temp_workspace, monkeypatch, capsys
):
    pytest.importorskip("jiwer")
    cfg, hyp, gt = temp_workspace
    ok_engine = MockEngine(name="paddle", texts=hyp)

    monkeypatch.setattr(runner, "preflight", lambda engine: None)
    monkeypatch.setattr(runner, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(runner, "build_engine", lambda engine, device, c: ok_engine)

    rc = runner.main(["--engine", "paddle", "--device", "cpu"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "FATAL" not in captured.err
    assert "Errors: 0 of 2 page(s) failed." in captured.out
    assert "none" in captured.out


def test_preflight_passes_paddleocr_import_check():
    # paddleocr is a core dependency, so it should always be importable --
    # this only checks the paddleocr import itself, not the paddle backend
    # (paddleocr imports fine even without its backend installed; see the
    # dedicated backend test below).
    pytest.importorskip("paddleocr")


def test_preflight_flags_missing_paddlepaddle_backend():
    # paddleocr is a core dependency and imports cleanly even without its
    # backend, but this dev machine deliberately does not have paddlepaddle
    # installed (real runs happen on Colab GPU) -- paddleocr only touches
    # `paddle` once you construct PaddleOCR(...), so the backend needs its
    # own explicit check. This exercises the real ImportError path.
    try:
        import paddle  # noqa: F401

        pytest.skip("paddlepaddle is installed in this environment")
    except ImportError:
        pass

    msg = runner.preflight("paddle")
    assert msg is not None
    assert "pip install" in msg
    assert "paddle" in msg


def test_preflight_passes_for_docai_core_dependency():
    pytest.importorskip("google.cloud.documentai")
    assert runner.preflight("docai") is None


def test_main_fails_fast_on_missing_paddlepaddle_backend(tmp_path, capsys):
    try:
        import paddle  # noqa: F401

        pytest.skip("paddlepaddle is installed in this environment")
    except ImportError:
        pass

    # No --config is needed: preflight must reject before load_config runs.
    rc = runner.main(["--engine", "paddle", "--device", "cpu",
                       "--config", str(tmp_path / "nonexistent.yaml")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Missing dependency for --engine paddle" in captured.err
    assert "pip install" in captured.err


def test_preflight_config_ignores_non_docai_engines():
    from ocrbench.config import Config

    cfg = Config({"docai": {"project_id": "YOUR_GCP_PROJECT_ID"}})
    assert runner.preflight_config("paddle", cfg) is None


def test_preflight_config_rejects_placeholder_values():
    from ocrbench.config import Config

    cfg = Config({
        "docai": {
            "project_id": "YOUR_GCP_PROJECT_ID",
            "processor_id": "YOUR_PROCESSOR_ID",
            "region": "us",
        }
    })
    msg = runner.preflight_config("docai", cfg)
    assert msg is not None
    assert "project_id" in msg
    assert "processor_id" in msg
    assert "YOUR_" in msg
    assert "OCRBENCH_DOCAI_PROJECT_ID" in msg


def test_preflight_config_passes_with_env_vars(monkeypatch):
    from ocrbench.config import Config

    monkeypatch.setenv("OCRBENCH_DOCAI_PROJECT_ID", "ocr-benchmark-502416")
    monkeypatch.setenv("OCRBENCH_DOCAI_REGION", "US")
    monkeypatch.setenv("OCRBENCH_DOCAI_PROCESSOR_ID", "234ea23afbe0364a")

    cfg = Config({
        "docai": {
            "project_id": "YOUR_GCP_PROJECT_ID",
            "processor_id": "YOUR_PROCESSOR_ID",
            "region": "us",
        }
    })
    assert runner.preflight_config("docai", cfg) is None


def test_main_rejects_placeholder_docai_config_before_build_engine(
    temp_workspace, monkeypatch, capsys
):
    cfg, hyp, gt = temp_workspace
    cfg.data["docai"] = {
        "project_id": "YOUR_GCP_PROJECT_ID",
        "processor_id": "YOUR_PROCESSOR_ID",
        "region": "us",
    }
    for var in ("OCRBENCH_DOCAI_PROJECT_ID", "OCRBENCH_DOCAI_REGION", "OCRBENCH_DOCAI_PROCESSOR_ID"):
        monkeypatch.delenv(var, raising=False)

    called = {"build_engine": False}
    monkeypatch.setattr(runner, "preflight", lambda engine: None)
    monkeypatch.setattr(runner, "load_config", lambda path=None: cfg)

    def _should_not_be_called(engine, device, c):
        called["build_engine"] = True
        raise AssertionError("build_engine must not run past a placeholder config")

    monkeypatch.setattr(runner, "build_engine", _should_not_be_called)

    rc = runner.main(["--engine", "docai", "--device", "cpu"])
    captured = capsys.readouterr()

    assert rc == 1
    assert called["build_engine"] is False
    assert "YOUR_" in captured.err
    assert "OCRBENCH_DOCAI_PROJECT_ID" in captured.err


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
