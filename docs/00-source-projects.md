# Source Projects: What `detect-mech-issues` and `detect-study` Are

This document explains the two sibling projects that `car-diagnosis` is extracted
from. Neither is the thing we are shipping — they are the quarry. `car-diagnosis`
takes the **sound pipeline** out of them, cleans it up, and ships it as a focused
open-source project.

---

## `detect-mech-issues` — the research project

**One line:** build a corpus and an ML model that identifies a car's mechanical
fault *from the sound it makes*.

This is where the hard research happened. The central question it set out to
answer was an honest one: **how much can you actually tell about a car fault from
audio alone?** The answer it arrived at — backed by leakage-safe evaluation — is
"less than you'd hope, but a useful amount if you're disciplined about it."

What it contains:

- **`youtube/`, `tiktok/`, `reddit/`** — three scraping + labeling pipelines, one
  per platform. Each downloads media, finds the mechanical-sound segments, strips
  out talking/music/silence, and attaches whatever text labels the platform offers
  (YouTube chapters/transcripts/comments, TikTok burned-in captions via OCR,
  Reddit post titles + top comments).
- **`shared/`** — cross-platform label refinement: multi-signal fusion (an LLM
  reasons over audio + text signals to infer a cause), code-side trust tiering
  (gold/silver/bronze), and a music-contamination filter.
- **`training1/`** — 37 scripts that turn the corpus into leakage-safe train/val/
  test splits, embed every clip with CLAP, train linear classifier heads, and
  calibrate confidence. This is the part with the rigorous evaluation.
- **`scraper/`, `napa/`, `inventory-browser/`** — a *separate* effort: a
  Camoufox-based scraper for auto-parts prices and labor estimates (RepairPal,
  NAPA, AutoZone), stored in SQLite and browsed via a React app. This answers
  "what would the repair cost?" and is **out of scope** for `car-diagnosis`.
- **`external-data/`** — fetch scripts for public reference datasets (Kaggle
  car-diagnostics, IDMT engine, DCASE, MIMII) used as verified evaluation anchors.

**Corpus reached:** ~12,600 clips (9,000 YouTube + 3,600 TikTok), of which ~2,100
are "gold" (audio and text independently agree on the cause).

**Key honest findings (these shape the product):**

- *Sound type* (grind / squeal / knock / whine / tick…) is decidable from audio.
  *Cause/part* (wheel bearing vs CV joint vs belt) mostly is **not** — it comes
  from text, not the waveform.
- Fine-grained part classification from audio alone is near chance. The one
  genuinely separable distinction is **engine-internal vs running-gear**, and the
  honest product is that coarse triage plus a *calibrated* confidence band.
- "Trust" must come from multiple signals agreeing, not from a model's
  self-reported confidence (the LLM systematically overstates its confidence).

## `detect-study` — the teaching / portfolio harness

**One line:** a 48-hour, full-stack rebuild that wraps the sound model in a
realistic product to demonstrate every layer end-to-end.

Where `detect-mech-issues` is the messy research lab, `detect-study` is the clean
reconstruction. It reorganized the sound code into a single, well-structured
`sound-diagnostics/` package and then built a product around it:

- **`sound-diagnostics/`** — the **consolidated** sound pipeline:
  `ingest/{youtube,tiktok,reddit}/` → `pipeline/` (fusion, tiering, music gate) →
  `training/{prep,features,models,eval}/` → `app/predict.py` (the inference entry
  point). **This is the cleanest version of the pipeline and the basis for
  `car-diagnosis`.**
- **`backend/`** — FastAPI + a Pydantic-AI agent that runs a multi-turn diagnostic
  Q&A grounded by the sound diagnosis + a repair order.
- **`extension/`** — a Chrome side-panel extension that scrapes a shop site and
  uploads a sound clip.
- **`fake-site/`, `mobile/`, `inventory-browser/`** — a fake shop site to scrape,
  a React Native app, and the parts browser.

Everything except `sound-diagnostics/` is **out of scope** for `car-diagnosis` —
that's the product wrapper, not the sound engine.

---

## What `car-diagnosis` takes from each

| Concern | Source of truth |
|---|---|
| Pipeline structure (`ingest`/`pipeline`/`training`/`app`) | `detect-study/sound-diagnostics/` |
| Scraping + cleaning + OCR/transcription logic | both (detect-study is cleaner) |
| Camoufox stealth-browser wrapper | `*/scraper/camoufox_scraper/browser.py` |
| Rigorous evaluation + calibration scripts | `detect-mech-issues/training1/` |
| Honest framing of what's achievable | `detect-mech-issues/docs/STATUS.md` |

**What it deliberately leaves behind:** the parts-price scraper, the SQLite parts
datastore, the inventory browser, the Pydantic-AI agent, the Chrome extension, the
fake site, and the mobile app. `car-diagnosis` is **only** scrape → clean →
segment → label → train → classify.
