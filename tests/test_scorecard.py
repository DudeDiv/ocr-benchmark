from ocrbench.scorecard import build_scorecard, _resource_index

WEIGHTS = {
    "accuracy": 0.4, "latency": 0.2, "cost": 0.15,
    "throughput": 0.15, "resources": 0.1,
}


def _agg(engine, device, wer, infer, cost, pps, ram=None):
    return {
        "engine": engine,
        "device": device,
        "mean_wer": wer,
        "mean_inference_seconds": infer,
        "cost_per_page": cost,
        "pages_per_second": pps,
        "peak_ram_mb": ram,
        "peak_cpu_percent": None,
        "peak_gpu_util_percent": None,
        "peak_vram_mb": None,
        "pages": [],
    }


def test_more_accurate_faster_engine_wins():
    paddle = _agg("paddle", "gpu", wer=0.10, infer=0.5, cost=0.0, pps=2.0, ram=2000)
    docai = _agg("docai", "cpu", wer=0.25, infer=1.5, cost=0.0015, pps=0.67)

    sc = build_scorecard([paddle, docai], WEIGHTS)
    assert sc["winner"] == "paddle_gpu"
    # scores are between 0 and 1
    for e in sc["engines"]:
        assert 0.0 <= e["total_score"] <= 1.0


def test_normalization_best_gets_one():
    a = _agg("paddle", "gpu", wer=0.10, infer=0.5, cost=0.0, pps=2.0, ram=1000)
    b = _agg("docai", "cpu", wer=0.30, infer=2.0, cost=0.0015, pps=0.5)
    sc = build_scorecard([a, b], WEIGHTS)
    by = {e["label"]: e for e in sc["engines"]}
    # paddle has the lowest WER -> normalized accuracy 1.0
    assert by["paddle_gpu"]["dimensions"]["accuracy"]["normalized"] == 1.0
    # docai has the worst WER -> normalized accuracy 0.0
    assert by["docai_cpu"]["dimensions"]["accuracy"]["normalized"] == 0.0


def test_missing_resources_scored_as_least_burden():
    # docai reports no local resource peaks -> should score 1.0 on resources.
    idx = _resource_index(_agg("docai", "cpu", 0.2, 1.0, 0.0015, 1.0))
    assert idx is None
    a = _agg("paddle", "gpu", wer=0.1, infer=0.5, cost=0.0, pps=2.0, ram=2000)
    b = _agg("docai", "cpu", wer=0.2, infer=1.0, cost=0.0015, pps=1.0)
    sc = build_scorecard([a, b], WEIGHTS)
    by = {e["label"]: e for e in sc["engines"]}
    assert by["docai_cpu"]["dimensions"]["resources"]["normalized"] == 1.0


def test_single_engine_ties_at_best():
    a = _agg("paddle", "cpu", wer=0.2, infer=1.0, cost=0.0, pps=1.0, ram=1500)
    sc = build_scorecard([a], WEIGHTS)
    assert sc["engines"][0]["total_score"] == 1.0
