<div align="center">
  <h1>cardiag</h1>
  <p><strong>Diagnose a car fault from its sound — a calibrated triage aid that says "uncertain" instead of bluffing</strong></p>

  [![ci](https://github.com/adam-s/car-diagnosis/actions/workflows/ci.yml/badge.svg)](https://github.com/adam-s/car-diagnosis/actions/workflows/ci.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
  [![AUROC 0.79](https://img.shields.io/badge/fault_AUROC-0.79_·_calibrated-4c72b0.svg)](docs/MODEL_CARD.md)

  [CLAP, step by step](https://adamsohn.com/clap/) · [Isolating the Engine Audio](https://adamsohn.com/separate/) · [Model Card](docs/MODEL_CARD.md) · [Defense](docs/DEFENSE.md)
</div>

---

`cardiag` is an end-to-end audio-ML pipeline. It scrapes fault-sound clips from
YouTube/TikTok, cleans the audio (isolating the mechanical sound from speech, music,
and noise), embeds it with a frozen CLAP model, and trains small linear heads to
triage the fault. It is exposed as a CLI and a live web app.

https://github.com/user-attachments/assets/23dc48e7-5571-4890-9a6c-4552d907dd8a

This is a proof of concept, and honest about what that means. Diagnosing a car fault
from a phone recording is genuinely hard, so `cardiag` is built as a calibrated
triage aid rather than a diagnoser: it tells you whether something sounds wrong,
roughly *where in the car* it is, and a ranked shortlist of likely parts. When the
audio won't support a call, it says "uncertain" instead of bluffing.

> The real contribution is the cleaning + honest-training recipe, which is reusable
> on other audio datasets. The modest accuracy here reflects how hard the problem is
> from crude phone audio (we hit the literature ceiling); the *same* method reaches
> 0.93 AUROC on clean engine audio. See [docs/DEFENSE.md](docs/DEFENSE.md).

## Interactive demos

Two pages visualize the first two stages of the pipeline:

- [Isolating the Engine Audio](https://adamsohn.com/separate/) — an interactive look at the `clean()` cascade pulling a short mechanical span out of noisy YouTube audio (speech, music, road noise).
- [CLAP, step by step](https://adamsohn.com/clap/) — how the frozen CLAP model turns those spans into the 512-d embedding the linear heads classify.

## What it actually achieves

Measured out-of-sample, leakage-safe (by-video grouped CV over 1,031 video groups;
permutation **p = 0.0005**). These are honest numbers, not a leaderboard.

| Capability | Result | vs. chance |
|---|---|---|
| Is something wrong? (fault/normal) | **AUROC 0.79** [0.76, 0.83] | 0.50 |
| Where in the car? (6 zones) | right zone in **top-3 ≈ 75%** | 2× |
| Which part? (12+ families) | right part in **top-3 ≈ 45–65%** | 3–4× |
| Knows when it doesn't know | calibrated (ECE ≈ 0.04), returns `UNCERTAIN` | — |

Full details, and the one head we *demoted* for failing out-of-sample (knock), are
in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Quickstart: clone to inference

A fresh clone is immediately usable. A small pre-trained model ships in `models/`,
and a synthetic demo clip is bundled, so nothing needs to be downloaded or scraped.

```bash
git clone <this-repo> && cd car-diagnosis
uv venv && source .venv/bin/activate
uv pip install -e ".[scrape,web,dev,viz]"     # Python 3.11

cardiag doctor                 # preflight: what's installed
cardiag train --fixtures       # a working model offline in ~2s (no scrape, no 2 GB download)
cardiag diagnose <clip.wav>    # verdict + where-in-the-car + ranked parts
cardiag serve --model models   # live web app: drop a clip / paste a link, "explain why"
```

Verify the whole thing end-to-end in an isolated worktree: `bash scripts/clone_verify.sh`.

## How it works

```text
audio ──► clean() cascade ──► CLAP embedding ──► linear heads ──► Diagnosis
          (isolate spans)     (frozen, 512-d)    (fault/region/    (calibrated,
                                                  part/knock)       UNCERTAIN-aware)
```

There is **one segmentation path**. Scraped clips, your own recordings (`cardiag
ingest`, any length), and uploads at inference all flow through the same `clean()`
cascade that isolates short mechanical spans. Spans over ~10 s are split into windows
so CLAP never silently truncates them. Training and serving share one embedding
contract, so there is no train/serve skew.

## Usage

```bash
cardiag diagnose clip.wav            # full model: verdict + region + ranked parts
cardiag triage   clip.wav            # calibrated engine-vs-running-gear
cardiag clean    clip.wav            # isolate the mechanical sound (no model needed)
cardiag inspect  clip.wav -o r.html  # SEE/HEAR the pipeline: spans, spectrograms, scores
cardiag ingest   ./my_audio --kind fault --cause wheel_bearing   # bring your own audio
cardiag scrape   youtube|tiktok      # build a corpus (Reddit is deprecated — too noisy)
cardiag train                        # train on your corpus
```

Add `--json` to any inference command for machine-readable output.

## Documentation

- [docs/DEFENSE.md](docs/DEFENSE.md) — the honest case that a deliberately crude method earns a real triage result.
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — per-head metrics, intended use, limitations.
- [docs/architecture.md](docs/architecture.md) — pipeline diagrams.
- [docs/scraping-guide.md](docs/scraping-guide.md) — start-to-finish corpus building.

## Scope & honesty

Valid for social-style / targeted-upload audio (YouTube, TikTok, or a phone clip a
user records deliberately). It is **not** a safety-critical or standalone diagnostic.
It is a triage assistant that narrows where to look and is honest about its
uncertainty. Model files are joblib artifacts: load only ones you trust.

License: see [LICENSE](LICENSE).
