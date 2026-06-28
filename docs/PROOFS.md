# Verification & Proofs

## Fresh-clone acceptance test (git worktree, no data, no model)

The headline proof: a developer who clones this with **nothing** can scrape →
clean → train → infer. Verified in an isolated `git worktree` containing only
committed code — no `data/`, no `.venv`, no model artifact — with a brand-new
virtualenv installed from `pyproject.toml`:

```
[3] confirm clean slate          model exists? False
[4] test suite                   36 passed, 3 deselected
[5] cardiag demo (from scratch)  scrape → clean → train → diagnose → exit 0
    ✓ loop complete: scraped, cleaned, trained, and diagnosed from scratch.
[6] diagnose w/ freshly-trained  verdict=fault, engine_internal p=0.748
```

Captured in [`proofs/worktree_fresh_clone.log`](../proofs/worktree_fresh_clone.log).
The suite also **skips** (not errors) when an optional extra is absent: with
`fastapi` uninstalled the web tests report `4 passed, 3 skipped`. Reproduce:

```bash
git worktree add --detach /tmp/clone-test HEAD
cd /tmp/clone-test && uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[scrape,web,dev]"
pytest && cardiag demo
```

The loop is self-contained — only `yt-dlp` + CLAP (no LLM, no external datasets,
no API keys). The demo model is intentionally small (a few videos); scrape more
(`cardiag scrape youtube --per-query 5 --max-videos 200`) for a stronger one.

---

This document is the honest accounting of what is tested, what was *run live*, and
what is — by the nature of the system — not provable to 100%. It is deliberately
candid: parts of this pipeline are deterministic and provable; parts depend on
live external sites; and the core is a probabilistic model whose "correctness" is
*measured accuracy + calibration*, not a boolean.

## The three kinds of "step" and the standard that applies to each

| Layer | Provable? | Standard applied here |
|---|---|---|
| Deterministic code (types, config, path translation, cause canonicalization, cascade math, Reddit parser) | **Yes (~100%)** | Unit tests on fixtures + synthetic signals |
| Inference wiring (clean → embed → heads → calibrated result) | **Yes** | Contract tests with a fixture model; live test with the real model |
| Live scraping (YouTube / TikTok / Reddit) | **No** (sites change) | Live transport smoke + pure-parser unit tests |
| The ML model itself | **No** (probabilistic) | *Measured* accuracy + calibration (ECE), not asserted |

## Automated test pyramid

Run `pytest` (the default deselects `live` and `e2e`).

| Suite | Count | What it proves | Needs |
|---|---|---|---|
| `test_unit_core.py` | 13 | types/enums, config invariants, `resolve_clip`, cause canon, Reddit `media_id`/`is_video` | nothing |
| `test_audio_cascade.py` | 6 | cascade isolates a loud span, finds nothing in silence; `clean()` on synthetic audio | silero-vad, librosa |
| `test_inference_contract.py` | 5 | `Classifier`/`Triage` wiring, verdict thresholds, JSON round-trip | fixture model (no CLAP) |
| `test_cli_web.py` | 7 | CLI commands + errors; web index/health/diagnose endpoint (web tests skip if `[web]` absent) | typer, fastapi |
| `test_build_labels.py` | 5 | from-scratch label logic (kind/knock/cause/triage) + head fitting | nothing |
| `test_live_model.py` | 2 | **real** model + CLAP diagnoses a real clip | model + CLAP (`-m live`) |
| `test_web_e2e.py` | 1 | **browser** upload → result renders (Playwright) | chromium (`-m e2e`) |

**Last run:** `36 passed, 3 deselected` (default) · `2 passed` (`-m live`) ·
`1 passed` (`-m e2e`) — **39/39 green**, including in the fresh worktree above. CI
([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs the default suite +
the Playwright e2e on every push.

## Live runs actually executed (on the author's M3, 2026-06-28)

These are real executions against the real model / real network, captured under
[`proofs/`](../proofs/) (gitignored artifacts; logs summarized here).

### 1. Inference on M3 / MPS with the real trained model — `proofs/inference_proof.txt`
`cardiag diagnose` / `triage` / `clean` run end-to-end on 7 real YouTube + TikTok
clips using `data/training/best_model_clap.joblib` on the Metal device. Highlights:
- The cleaning cascade **isolated the actual sound** from a longer YouTube clip
  (kept 12.99–14.25s and 14.79–15.84s, dropped the rest) — the "10-minute video,
  ~15 seconds of real sound" case.
- Graceful **whole-clip fallback** when no distinct span survives.
- `triage` returns a calibrated band; `clean` reports `music_probability` + `is_music`.

### 2. Showcase across sources & situations — `proofs/showcase.jsonl`
`scripts/showcase.py` sampled **270 real clips** (90 each from YouTube / TikTok /
Reddit) and ran the full `clean()` + `diagnose()` on every one in **38 s** (~0.16
s/clip on MPS). It covers exactly the hard cases:

| Situation | Count | Meaning |
|---|---|---|
| `whole_clip` | 124 | nothing distinct to isolate → graceful fallback |
| `isolated` | 63 | a sub-span isolated from a longer clip (the "15s of a 10-min video") |
| `clean_full` | 44 | the whole clip is the mechanical sound |
| `compilation` | 20 | multiple distinct spans (a compilation, broken up) |
| `music_skipped` | 19 | music gate flagged it (the "TikTok that's only music") |

Verdict split: 132 fault / 82 normal / 56 uncertain.

### 3. Labeling stage — `proofs/labeling_proof.txt`
The swappable LLM backend ([`cardiag.pipeline.llm`](../src/cardiag/pipeline/llm.py)):
- **Ollama `qwen2.5:7b` (local, $0, on M3):** canonicalized 3 scraped titles to
  `wheel_bearing` / `belt` / `cv_joint` in 5.9 s.
- **Sonnet via `claude -p --model claude-sonnet-4-6`:** fused audio + OCR + comment
  signals into a structured `{fused_cause, fused_kind, fused_confidence, support}`
  in 12.3 s — the multi-signal fusion the corpus was built with.

### 4. Scrape transports (live) — `proofs/scrape_proof.txt`
- **YouTube:** `yt-dlp ytsearch` returned 5 on-target videos in 3.1 s.
- **Reddit:** the ported `list_page()` parsed **25 live posts** (6 with video) from
  `old.reddit.com/r/MechanicAdvice` in 1.5 s.
- **TikTok:** discovery uses a stealth browser (patchright) + network interception;
  not run here (needs a browser + anti-bot), but the code is
  [`cardiag.ingest.tiktok.discover`](../src/cardiag/ingest/tiktok/discover.py).

## What is deliberately NOT claimed

- **Not 100% correct.** Fine cause from audio alone has a measured ceiling (~0.33
  top-1); the honest products are fault-vs-normal (~0.74) and the calibrated
  engine-vs-running-gear triage (ECE ≈ 0.018). See [research/STATUS.md](research/STATUS.md).
- **Scrapers will drift** as the platforms change their markup/APIs. The parsers are
  unit-tested and the transports were smoke-tested live, but live scraping is never
  "proven forever."
- **The full corpus/model is not regenerated here.** The 12,600-clip corpus + full
  training take hours and ~13 GB; these proofs run each *stage* on a real sample and
  reuse the already-trained model artifact. Reproduce the full run with
  `cardiag scrape …` then `cardiag train`.

## Reproduce the proofs

```bash
uv pip install -e ".[web,dev,scrape]"
# inference + showcase need a model in data/training/ (cardiag train, or copy one)
python scripts/showcase.py 90                 # -> proofs/showcase.jsonl
cardiag diagnose proofs/<clip>.wav --json     # -> a Diagnosis
pytest && pytest -m live && pytest -m e2e      # the full pyramid
```
