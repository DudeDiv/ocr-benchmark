# ocrbench

I built this to answer a fairly specific question: for scanned handwritten UPSC
CSE answer booklets, is PaddleOCR good enough, or is it worth paying for Google
Document AI? Eyeballing a few pages wasn't cutting it, so ocrbench runs both
engines over the same pages and reports the numbers that actually matter to me —
word/character error rate, how long each page takes, what it costs, throughput,
and how hard PaddleOCR leans on the machine. At the end it boils all of that down
into a single weighted score so I can compare the two at a glance.

## Getting set up

```bash
pip install -e .
```

That pulls in everything both engines' Python code needs — `google-cloud-documentai`
and `paddleocr` are both normal, cross-platform packages, so they're core
dependencies, not extras. GPU resource sampling and dev tooling are the only
optional extras, so grab whichever you need:

```bash
pip install -e ".[gpu]"     # pynvml, for reading GPU util/VRAM
pip install -e ".[dev]"     # pytest
```

There's still one manual step for Paddle:

```bash
pip install paddlepaddle-gpu         # or paddlepaddle for CPU-only
```

`paddlepaddle`/`paddlepaddle-gpu` is the actual inference backend paddleocr
calls into, and it's deliberately **not** listed anywhere in `pyproject.toml` —
verified with `pip install --dry-run paddleocr` that installing paddleocr alone
never pulls it in transitively, which is by design: it has to match the CUDA
version of whatever machine actually runs it (mine is a Colab GPU runtime), so
pinning a version here would just be wrong somewhere else. Check [the
PaddlePaddle install matrix](https://www.paddlepaddle.org.cn/en/install/quick)
for the exact command for your CUDA version. Everything else — the whole rest
of the package, including paddleocr itself — imports and tests fine without it,
so you can develop locally regardless.

If you run `ocrbench.run --engine paddle` without paddlepaddle installed (or
`--engine docai` without google-cloud-documentai, if you've somehow pruned it),
the CLI checks for it up front and tells you the exact `pip install` to run,
rather than dying on the first page — or twelve pages in.

## How it's laid out

```
config.yaml            # paths, DPI, Doc AI processor, manifests, weights, boilerplate
ocrbench/
  config.py            # loads config.yaml and resolves paths
  preprocess.py        # PyMuPDF renders manifest pages -> work/images/{doc}/page_{n}.png
  engines/
    base.py            # the OCREngine interface + OCRResult/Word
    paddle_engine.py   # PaddleOCR 3.x (English, device=cpu/gpu, warm-up call isn't timed)
    docai_engine.py    # Document AI, keeps the raw JSON under work/raw/docai/
  metrics.py           # text normalization + WER/CER via jiwer, plus confidence stats
  resources.py         # background psutil/pynvml sampler, only runs during Paddle
  run.py               # the CLI: writes results/{engine}_{device}.json + a combined CSV
  scorecard.py         # turns the results into one weighted score
tests/                 # unit tests against a mocked engine (no paddle required)
```

## Running a benchmark

Drop the source booklet PDFs in `pdfs/` at the repo root — one file per document
id, e.g. `pdfs/GS-1.pdf` for the `GS-1` manifest entry. The lookup is tolerant:
it matches `{doc}*.pdf` case-insensitively, so `gs-1.pdf` or `GS-1_scan.pdf` both
work. If nothing matches, it fails with an error naming the doc id, the
directory it searched, and the files it actually found there — no silent
mismatch.

Three steps. First render the pages you care about to images, then run each
engine, then look at the scorecard:

```bash
# 1. render the manifest pages to PNGs
python -m ocrbench.preprocess

# 2. run each engine/device combo you want to compare
python -m ocrbench.run --engine paddle --device gpu
python -m ocrbench.run --engine docai  --device cpu

# 3. build the scorecard from everything in results/
python -m ocrbench.scorecard
```

For Document AI you'll need to point it at your processor: `project_id`,
`region`, and `processor_id`. `config.yaml` only ever ships placeholders for
these (`YOUR_GCP_PROJECT_ID`, etc.) — I don't want the real project/processor
ids sitting in version control, same reasoning as the credentials below. The
real values are meant to come from the environment:

```bash
export OCRBENCH_DOCAI_PROJECT_ID=your-real-project-id      # bash
export OCRBENCH_DOCAI_REGION=us                             # "us" or "eu"
export OCRBENCH_DOCAI_PROCESSOR_ID=your-real-processor-id
```

Environment variables win; `config.yaml` is only a fallback. If you'd rather not
use env vars, you can edit the `docai:` section of `config.yaml` directly — just
know it'll be sitting in the repo from then on. Either way, a leftover
`YOUR_...` placeholder is rejected at startup with a clear message, rather than
surfacing as an opaque `403 CONSUMER_INVALID` from the API mid-run.

Credentials are a separate thing and never go in the config either — they come
from the environment so the key file doesn't end up in version control:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json          # bash
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"      # PowerShell
```

The runner never ends quietly. Every run prints an error summary by exception
type (even "none" if everything succeeded), and if every single page fails it
still writes the results file — so the per-page error details are there to
inspect — but also prints the first three tracebacks and exits non-zero, rather
than reporting success on a run that produced nothing.

## About the manifests

The booklets aren't all answers — there are cover sheets, instruction pages, and
examiner feedback pages mixed in, and scoring those would just be noise. So each
document lists exactly which (1-indexed) PDF pages are real answer pages:

```yaml
manifests:
  doc001: [3, 4, 5]
```

Drop the corresponding ground truth in `ground_truth/{doc}/page_{n}.txt` and the
metrics will pick it up automatically. Ground truth is optional per page — if a
`page_N.txt` doesn't exist, that page's `wer`, `cer`, and `exact_match` come back
as null and the run keeps going. Only the pages you've actually transcribed get a
real error rate.

## How the accuracy numbers hold up

Hand-transcribing booklets is slow, so only a subset of pages has verified ground
truth. Those pages are the real evidence — the WER/CER you can defend. To avoid
resting the whole accuracy story on a dozen pages, every page (transcribed or
not) also gets two reference-free proxies:

- **`dictionary_validity`** — the fraction of alphabetic output words that are
  real English words (via pyspellchecker). It catches garbage output, but it
  can't see valid-word substitutions: "cat" mis-read as "cot" still scores as
  valid. So it tells you the OCR is producing words, not that they're the right
  words.
- **`cross_engine_agreement`** — WER between the PaddleOCR and Doc AI outputs for
  the same page after normalization. It shows you where the two engines diverge,
  which is where to look — but agreement isn't correctness. If both engines make
  the same mistake it reads as perfect agreement, and a high value tells you they
  disagree, not who's right. (This one only fills in once both engines have run.)

Treat these as supporting signals with those limits stated plainly. The verified
pages anchor the real WER/CER; the proxies just extend the pattern across the
full set.

## Tests

```bash
pytest
```

The tests run against a mocked engine, so you don't need PaddleOCR or a Google
Cloud account to run them. Anything that genuinely needs `jiwer` will skip itself
if it isn't installed.
