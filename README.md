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

That gets you the core package. The engine-specific and GPU bits are optional
extras, so grab whichever you need:

```bash
pip install -e ".[docai]"   # google-cloud-documentai
pip install -e ".[gpu]"     # pynvml, for reading GPU util/VRAM
pip install -e ".[dev]"     # pytest
```

One thing worth calling out: `pip install -e .` does **not** pull in PaddleOCR.
That's on purpose. paddlepaddle is a pain to install locally (and I don't want it
on my laptop), so I only install `paddlepaddle`/`paddleocr` on the box that does
the real inference — usually a Colab GPU runtime. Everything still imports fine
without it, so you can develop and run the tests locally either way.

## How it's laid out

```
config.yaml            # paths, DPI, Doc AI processor, manifests, weights, boilerplate
ocrbench/
  config.py            # loads config.yaml and resolves paths
  preprocess.py        # PyMuPDF renders manifest pages -> work/images/{doc}/page_{n}.png
  engines/
    base.py            # the OCREngine interface + OCRResult/Word
    paddle_engine.py   # PaddleOCR (English, GPU toggle, warm-up call isn't timed)
    docai_engine.py    # Document AI, keeps the raw JSON under work/raw/docai/
  metrics.py           # text normalization + WER/CER via jiwer, plus confidence stats
  resources.py         # background psutil/pynvml sampler, only runs during Paddle
  run.py               # the CLI: writes results/{engine}_{device}.json + a combined CSV
  scorecard.py         # turns the results into one weighted score
tests/                 # unit tests against a mocked engine (no paddle required)
```

## Running a benchmark

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
