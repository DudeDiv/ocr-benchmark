# ocrbench

Benchmark **PaddleOCR** vs **Google Document AI** on scanned handwritten English
exam booklets (UPSC CSE answer sheets). Measures accuracy (WER/CER), latency,
cost, throughput, and — for the local engine — peak resource usage, then rolls
everything into a weighted scorecard.

## Install

```bash
pip install -e .
# optional extras (install as needed):
pip install -e ".[docai]"   # google-cloud-documentai
pip install -e ".[gpu]"     # pynvml for GPU sampling
pip install -e ".[dev]"     # pytest
```

> **PaddleOCR is intentionally not installed by `pip install -e .`.** Install
> `paddlepaddle`/`paddleocr` only on the machine that runs real inference
> (e.g. Colab GPU). Every module imports fine without it.

## Layout

```
config.yaml            # paths, DPI, Doc AI processor, manifests, weights, boilerplate
ocrbench/
  config.py            # config loader + path helpers
  preprocess.py        # PyMuPDF: render manifest pages -> work/images/{doc}/page_{n}.png
  engines/
    base.py            # OCREngine ABC, OCRResult, Word
    paddle_engine.py   # PaddleOCR (lang=en, GPU flag, warm-up excluded from timing)
    docai_engine.py    # Document AI process_document, saves raw JSON -> work/raw/docai/
  metrics.py           # normalize -> WER/CER (jiwer) + confidence stats
  resources.py         # psutil + pynvml sampler thread (Paddle runs only)
  run.py               # CLI: results/{engine}_{device}.json + combined CSV
  scorecard.py         # weighted scorecard across the five dimensions
tests/                 # mocked-engine unit tests (no paddle needed)
```

## Usage

```bash
# 1. Render the manifest pages to PNGs
python -m ocrbench.preprocess

# 2. Run a benchmark (repeat per engine/device)
python -m ocrbench.run --engine paddle --device gpu
python -m ocrbench.run --engine docai  --device cpu

# 3. Build the scorecard from all results/*.json
python -m ocrbench.scorecard
```

Document AI credentials come from the environment, never `config.yaml`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json   # bash
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"  # PowerShell
```

Set `docai.project_id`, `docai.processor_id`, and `docai.region` in `config.yaml`.

## Manifests

Booklets contain cover, instruction, and feedback pages that must not be scored.
`config.yaml` lists, per document, the 1-indexed PDF pages that are actual
answer pages:

```yaml
manifests:
  doc001: [3, 4, 5]
```

Ground truth goes in `ground_truth/{doc}/page_{n}.txt`.

## Tests

```bash
pytest
```

Tests use a mocked engine, so they run without PaddleOCR or Google Cloud. Tests
that need `jiwer` are skipped automatically if it is not installed.
```
