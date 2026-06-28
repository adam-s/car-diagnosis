# cardiag

**Diagnose a car's mechanical fault from the sound it makes.** A complete,
honest audio-ML pipeline: scrape fault-sound clips from YouTube / TikTok / Reddit,
clean and segment the audio, label it with OCR / transcript / text metadata, train
a calibrated classifier, and serve inference through one shared core that powers a
CLI and a local web app.

```python
from cardiag import Classifier

clf = Classifier.load()
result = clf.diagnose("noise.wav")     # cleans → embeds → heads → calibrated result
print(result.verdict)                  # Verdict.FAULT
print(result.fault_probability)        # 0.81
print(result.causes[0])                # Cause(part='wheel bearing', p=0.34, note='…')
```

```
$ cardiag diagnose noise.wav
  noise.wav
  ────────────────────────────────────────────────────
  Verdict: FAULT  (fault p=0.81)
  Most likely cause (audio is suggestive, not definitive):
     34% ███████              wheel bearing  worn hub/bearing — inspect
     22% ████                 brakes         worn brake pads/rotors
     14% ██                   cv joint       CV axle / joint
  Isolated 3 mechanical span(s), 6.2s total.
  ────────────────────────────────────────────────────
  Fine cause from sound alone is uncertain; treat as triage, not a final diagnosis.
```

---

## The honest thesis

This project is built around a result that took a 12,600-clip corpus and
leakage-safe evaluation to pin down:

> **Sound *type* (grind / squeal / knock / whine / tick) is decidable from audio.
> The *cause/part* mostly is not — it comes from text.**

So `cardiag` does two things well and refuses to pretend about the rest:

- **`diagnose`** — fault-vs-normal (~0.74 balanced accuracy on verified clips),
  engine-knock detection, and a *ranked, probabilistic* cause guess that is never
  presented as a single confident answer.
- **`triage`** — the one acoustically separable distinction, **engine-internal vs
  running-gear**, with a **calibrated** confidence band (ECE ≈ 0.018; HIGH band
  ~90% correct) so a human knows when to trust it. It abstains rather than guess.

That restraint is the point. An overconfident single-cause classifier would be
easy to build and wrong; this one is calibrated and says so.

## How it works

```
SCRAPE            CLEAN + SEGMENT          LABEL                TRAIN              CLASSIFY
YouTube  yt-dlp   energy → VAD →           OCR consensus        CLAP embed (512d)  clean → embed →
TikTok   Camoufox spectral flatness →      transcript/chapters  → linear heads     heads → calibrated
Reddit   old HTML music gate (CLAP)        Sonnet fusion+tier   (kind/knock/cause) Diagnosis
```

The cleaning cascade (`cardiag.audio.clean`) is the same code at corpus-build time
and at inference, so an uploaded clip is processed exactly like a training clip —
music, voice, and static stripped, leaving only the mechanical sound. See
[`docs/`](docs/) for the full design and Mermaid diagrams.

## Install

```bash
git clone https://github.com/adamsohn/car-diagnosis && cd car-diagnosis
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[web,dev]"     # add "scrape" for the corpus pipeline
```

Requires Python 3.11. A trained model (`best_model_clap.joblib`) is needed for
`diagnose`/`triage`; build one with `cardiag train` after scraping, or point
`CARDIAG_DATA` at an existing `data/training/` directory.

## Usage

```bash
cardiag diagnose clip.wav            # full model: verdict + knock + ranked causes
cardiag triage   clip.wav            # calibrated engine-vs-running-gear
cardiag clean    clip.wav            # isolate the mechanical sound (no model needed)
cardiag serve                        # local drag-and-drop web app at :8000
cardiag scrape   youtube|tiktok|reddit
cardiag train
```

Add `--json` to any inference command for machine-readable output.

## Package layout

```
src/cardiag/
  __init__.py        public API: Classifier, TriageClassifier, clean, Diagnosis
  config.py          one source of truth: sample rates, prompts, thresholds, taxonomy
  types.py           typed results: Diagnosis, Cause, Segment, TriageResult
  audio/             clap (wrapper) · cascade (cleaning) · clean (public)
  inference/         classifier (fault/knock/cause) · triage (calibrated)
  ingest/            youtube · tiktok · reddit  — scrape + corpus build
  pipeline/          fusion (LLM) · gate_tier · music_gate · llm (swappable backend)
  training/          prep · features · models · eval
  web/               FastAPI upload app + static UI
tests/               unit · contract · integration · e2e (Playwright)
docs/                design docs + research notes + verified-run log (PROOFS.md)
```

## Testing

```bash
pytest                       # unit + contract + integration smoke (fast, no model)
pytest -m model              # tests that need a trained model artifact
pytest -m e2e                # Playwright web end-to-end (needs: playwright install)
```

What is and isn't provable is documented honestly in
[`docs/PROOFS.md`](docs/PROOFS.md): deterministic logic is unit-tested to high
confidence; scrapers are tested against recorded fixtures (live sites change); and
model quality is *measured* (accuracy + calibration), not asserted.

## Acknowledgements

The pipeline and corpus methodology were developed in two predecessor research
projects; `cardiag` is the cleaned, focused, open-source extraction of the sound
engine. Built on CLAP (`laion/clap-htsat-unfused`), Silero VAD, librosa, yt-dlp,
Camoufox, and scikit-learn.

## License

MIT — see [LICENSE](LICENSE).
