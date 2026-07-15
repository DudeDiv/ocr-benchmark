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

That's enough for Doc AI — `google-cloud-documentai` is a normal, cross-platform
client library, so it's a core dependency, not an extra. GPU resource sampling
and dev tooling are optional extras, so grab whichever you need:

```bash
pip install -e ".[gpu]"     # pynvml, for reading GPU util/VRAM
pip install -e ".[dev]"     # pytest
```

PaddleOCR is the one exception, and it's a two-step install:

```bash
pip install -e ".[paddle]"           # paddleocr itself
pip install paddlepaddle-gpu         # or paddlepaddle for CPU-only
```

`paddlepaddle`/`paddlepaddle-gpu` is deliberately **not** listed anywhere in
`pyproject.toml`, extras included — it has to match the CUDA version of whatever
machine actually runs it (mine is a Colab GPU runtime), so pinning a version here
would just be wrong somewhere. Check [the PaddlePaddle install
matrix](https://www.paddlepaddle.org.cn/en/install/quick) for the exact command
for your CUDA version before installing it. Everything else — the whole rest of
the package — still imports fine without either paddle package installed, so you
can develop and run the tests locally regardless.

If you run `ocrbench.run --engine paddle` without paddleocr installed (or
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

For Document AI you'll need to point it at your processor. Set `project_id`,
`processor_id`, and `region` under `docai:` in `config.yaml`. Credentials, on the
other hand, never go in the config — they come from the environment so the key
doesn't end up in version control:

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
