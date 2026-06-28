# What We Have: Inventory of Reusable Components

`car-diagnosis` does not need to invent anything. Every stage already exists and
has been run at scale across `detect-mech-issues` and `detect-study`. This document
is the parts list — what exists, what it does, the exact library/model it uses, and
where to lift it from.

The scope of `car-diagnosis` is exactly four stages:

```
SCRAPE  →  CLEAN + SEGMENT  →  LABEL (OCR / text / transcription)  →  TRAIN  →  CLASSIFY
```

Everything below maps to one of those stages. The canonical layout to port to is
`detect-study/sound-diagnostics/`, which already organizes the code this way.

---

## 1. Scrape — acquire raw media + native metadata

Three platforms, three transports. None of them needs a paid API.

| Platform | Discovery | Download | Source files |
|---|---|---|---|
| **YouTube** | `yt-dlp` `ytsearch` over fault + normal queries | `yt-dlp` (audio only) | `ingest/youtube/discover.py`, `batch.py`, `capture.py` |
| **TikTok** | stealth browser intercepts `/api/search/item/full/` feed JSON | `yt-dlp` (mp4) | `ingest/tiktok/discover.py`, `batch.py` |
| **Reddit** | `old.reddit.com` HTML scrape (regex over structured attrs), paginated | `yt-dlp --cookies-from-browser firefox` pulls `v.redd.it` | `ingest/reddit/scrape.py`, `pipeline.py` |

Notes that matter:

- **YouTube** is the volume driver (~9k clips). `discover.py` runs curated fault
  queries ("car making grinding noise when braking") and matched *normal* queries
  (normal must be *earned*, not assumed).
- **TikTok** is the label driver — short clips with the fault name **burned into the
  video as text**. Discovery currently uses `patchright` (stealth Chromium) to
  intercept the search API; this is the one place we will standardize onto Camoufox
  (see [`02-scraping-with-camoufox.md`](02-scraping-with-camoufox.md)). Account
  catalogs are then expanded with `yt-dlp` (374 seed videos → 1,740).
- **Reddit** is robust and auth-light: the `.json` API 403s under load, but
  `old.reddit.com` HTML stays open. Extraction is deterministic regex over
  `data-fullname` / `data-url` / `data-permalink`. The only authenticated step is
  `yt-dlp` reusing Firefox session cookies to pull the actual audio. Restart-safe
  via a JSONL ledger; dedups reposts across subreddits by media id.

**Native metadata captured at scrape time** (this is the cheap, high-value text
signal — no transcription needed):

- YouTube: chapters, auto-captions/transcript, top comments (`yt-dlp
  --write-subs --write-auto-subs`, plus the comment JSON).
- TikTok: the post caption + hashtags.
- Reddit: post title, self-text, and top-8 comments (sorted by score, automod
  filtered).

## 2. Clean + Segment — find the mechanical sound, drop everything else

This is the "removal of music, voice, other sounds" the project description calls
for. It is a **cheap cascade** that runs CPU-first and only spends the expensive
model on survivors. Same code is reused for *both* corpus building and inference.

Source: `ingest/youtube/audio.py` (shared CLAP wrapper + cascade), `pipeline/music_gate.py`.

| Filter | Removes | Library / model | Why this one |
|---|---|---|---|
| **Energy / RMS gate** | silence, dead air | `librosa` RMS | trivially cheap, runs first |
| **Voice Activity Detection** | talking, narration | **Silero VAD** (1.8 MB, ~200× realtime) | beats `webrtcvad`, which misclassifies mechanical noise as speech |
| **Spectral flatness gate** | wind, hiss, static | `librosa` spectral flatness | rejects broadband noise that isn't a fault |
| **Music gate** | background music | **CLAP** zero-shot score over `[music, mechanical, speech, silence]`, threshold 0.5 | measured ~26% of TikTok was music; CLAP rejects it cleanly |

The cascade discards ~97% of talky video *before* the expensive CLAP confirmation
runs. What survives is a set of candidate audio **regions**; each is snapped to
transient boundaries and written out as a clip WAV (mono, resampled).

After cleaning, **CLAP confirmation + L1 labeling** runs on survivors only:

- **Model:** `laion/clap-htsat-unfused` via HuggingFace `transformers`
  (`ClapModel` / `ClapProcessor`), on MPS/CPU.
- Zero-shot scoring against prompt sets with **margin gating**: confirm the sound
  is mechanical (margin ≥ 0.50), discriminate fault-vs-shop-tool (≥ 0.25), then
  pick an L1 sound type (≥ 0.15; normal needs ≥ 0.30).
- Also computes cheap DSP features per clip: spectral centroid/flatness, and
  **cyclic features** (periodicity, pulse Hz, regularity) — a knock is periodic, a
  hiss is not.

## 3. Label — OCR, text, and transcription metadata

The pipeline's core insight: **the cause comes from text, not the waveform.** So
extracting good text per clip is as important as the audio.

| Source | Technique | Library | Files |
|---|---|---|---|
| **TikTok on-screen captions** | extract 3 frames per clip, OCR each, take a **temporal consensus** (recurring text wins; one-off narration is dropped) | **`ocrmac`** (Apple Vision, on-device Neural Engine — doesn't steal the GPU from CLAP) | `ingest/tiktok/batch.py` |
| **YouTube chapters / transcript / comments** | captured at scrape time, joined down to clip level by timestamp | `yt-dlp` | `ingest/youtube/capture.py`, `enrich.py` |
| **Reddit title + comments** | scraped HTML, top comments by score | regex | `ingest/reddit/scrape.py` |
| **Label canonicalization** | map raw/weak labels → canonical part families | Claude **Haiku** via swappable LLM backend | `ingest/*/normalize.py` |

On transcription: Whisper was evaluated and is **not used on the audio corpus** —
by construction these clips have no speech (VAD removed it). The "transcription"
signal is YouTube's own captions, captured for free at scrape time, not generated.

**OCR consensus detail (the part worth keeping):** single-frame OCR yielded ~28%
usable labels; 3-frame voting (rank by size × confidence, strip TikTok banners/
channel names) lifted it to ~50%. This is in `ingest/tiktok/batch.py`.

## 4. Fuse + Tier — turn signals into trustworthy labels

Before training, every clip's audio + text signals are fused into a single label
with an honest trust tier. Source: `pipeline/`.

- **`fusion.py`** — runs Claude **Sonnet** over each clip's signals (L1 sound type,
  rhythm, chapter, transcript, OCR, title, comment) and returns
  `{cause, kind, confidence, support}`. Hard rule: confidence ≥ 0.7 **requires
  corroborating text**; sound-type alone caps at 0.45. Parallel + resumable.
- **`gate_tier.py`** — assigns trust tier from *objective* text signals, not model
  confidence: **gold** (a fault cause with clean text corroboration), **silver**
  (fault, audio only), **bronze** (no cause / normal). Auto-rejects junk OCR.
- **`music_gate.py`** — final CLAP music pass; music-contaminated clips are excluded
  from gold.
- **`llm.py`** — swappable LLM backend that routes to the cheapest adequate option:
  local Ollama (Qwen2.5, $0) → Modal (Qwen GPU) → Claude (Haiku fallback). JSON-only
  prompts, parallel batching, error recovery.

## 5. Train — corpus → calibrated classifier

Source: `training/` (`prep/`, `features/`, `models/`, `eval/`), the cleaned-up
descendant of `detect-mech-issues/training1/`.

| Step | What it does | Library | File |
|---|---|---|---|
| **Prepare** | merge labels, build **leakage-safe** train/val/test splits grouped by creator/video | stdlib | `training/prep/prepare.py` |
| **Canonicalize causes** | 359 raw causes → ~24 part families | stdlib | `training/prep/causes.py` |
| **Embed** | frozen **CLAP** 512-d embeddings for every clip, cached to `.npz` (resumable) | `transformers`, `torch` | `training/features/embed.py` |
| **Train heads** | linear heads on frozen embeddings: `kind` (fault/normal), `knock` (knock/normal), `cause` (multi-class) | **scikit-learn** `LogisticRegression` + `StandardScaler`, saved via `joblib` | `training/models/train_best.py` |
| **Calibrate** | isotonic calibration → confidence bands (HIGH/MED/LOW/ABSTAIN); measure **ECE** | scikit-learn | `training/models/confidence.py`, `eval/calibrate.py` |
| **Evaluate** | creator-grouped CV, coverage-at-precision, system-level eval | scikit-learn | `training/eval/*.py` |

**Why frozen CLAP + linear heads (not fine-tuning):** it's the configuration that
the rigorous evaluation actually validated, it trains in seconds on CPU, and it
keeps the honest framing intact. (BEATs/AST embeddings are stubbed as a future
upgrade — `embed_ast.py` — but not required to ship.)

**Measured results to reproduce honestly in the README:**

- Fault-vs-normal: ~0.74 balanced accuracy on verified clips.
- Cause top-1 ~0.33 / top-3 ~0.59 in-domain (majority baseline ~0.19) — i.e. weak,
  text-dependent, *not* a confident single-cause answer.
- Engine-internal vs running-gear coarse triage with calibrated bands: HIGH band
  ~90% accurate, covering ~half of clips; ECE ~0.018.
- Knock recall ~0 in CLAP space — a documented feature ceiling, not a bug.

## 6. Classify — the abstracted inference component

This is the deliverable the project description asks for: **one inference core,
two front-ends (CLI and local web), and the uploaded clip is cleaned with the exact
same cascade used in training.**

The core already exists: `sound-diagnostics/app/predict.py`.

```
uv run app/predict.py <audio.wav> [--model data/best_model_clap.joblib] [--json]
```

It loads CLAP locally, embeds the clip, runs the three linear heads, and returns:

```json
{
  "verdict": "fault",
  "fault_probability": 0.81,
  "engine_knock_probability": 0.12,
  "top_causes": [{ "part": "wheel_bearing", "p": 0.34, "note": "..." }]
}
```

**The abstraction to build (small, and the only genuinely new wiring):**

```
app/clean.py     # the §2 cascade as a reusable function: wav -> [clean segment(s)]
app/predict.py   # core: clean -> CLAP embed -> heads -> calibrated result  (exists)
app/cli.py       # thin CLI front-end over predict()                         (exists as demo.py)
app/web.py       # local FastAPI: upload wav -> clean -> predict() -> result (new, ~1 file)
```

The web front-end is a single local FastAPI route plus a static upload page — it
reuses `predict()` unchanged. Critically, **both front-ends call the same
`clean()`** so an uploaded clip is music/voice/noise-stripped exactly as training
clips were. This is what makes inference match training instead of drifting.

---

## Dependency summary

Everything here is open and already in use:

- **Scraping:** `yt-dlp`, `camoufox` (≥ 0.4.11) + `playwright` (== 1.51.0, pinned to
  Camoufox's Firefox 135), `ffmpeg`, stdlib `urllib`.
- **Audio / cleaning:** `librosa`, `soundfile`, `silero-vad`, `ffmpeg`.
- **Models:** `transformers` (CLAP `laion/clap-htsat-unfused`), `torch`,
  `scikit-learn`, `joblib`.
- **OCR:** `ocrmac` (Apple Vision; macOS). For a cross-platform OSS fallback,
  `easyocr`/`paddleocr` slot into the same 3-frame-consensus interface.
- **LLM (label refinement only):** swappable — Ollama / Modal / Claude. Not required
  at inference time.
- **Inference front-ends:** `fastapi` + `uvicorn` (local web), stdlib `argparse` (CLI).

Python ≥ 3.11, managed with `uv`. Scripts use PEP-723 inline dependency headers so
each stage is independently runnable.
