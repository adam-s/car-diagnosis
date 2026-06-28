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

## Quickstart — clone to inference, from nothing

No clips and no model are bundled (this repo *teaches the loop by running it*). A
fresh clone builds its own model from scratch:

```bash
git clone https://github.com/adamsohn/car-diagnosis && cd car-diagnosis
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[scrape,web,dev,viz]"

cardiag doctor               # checks your setup and tells you exactly what's next
```

From here you never have to guess the next command — each one ends by pointing to
the next. `cardiag doctor` will hand you the gentle path:

```bash
cardiag train --fixtures     # a working model in ~2s, offline (no scrape, no 2GB download)
cardiag inspect some.wav     # SEE + HEAR what the pipeline does, then it nudges you onward
cardiag demo                 # the whole loop for real: scrape → clean → train → diagnose
```

`cardiag demo` discovers fault + normal videos (YouTube via yt-dlp, TikTok + Reddit
via Camoufox), cleans them, CLAP-embeds the clips, trains the models, and diagnoses
one — the whole pipeline on your machine in a few minutes. **No LLM, no external
datasets, no API keys.**

Then run it for real (more clips → a better model):

```bash
cardiag scrape youtube --per-query 5 --max-videos 200
cardiag train
cardiag diagnose some_clip.wav
```

Requires Python 3.11. (`docs/` describes the fuller research pipeline — LLM label
fusion + verified external anchors — that produces the higher-fidelity model.)

**Scraping transport.** [Camoufox](https://camoufox.com/) (stealth Firefox) is the
default browser for all page/feed scraping — **TikTok** search and **Reddit**
listings both go through it (`python -m camoufox fetch` once to install its
Firefox; `playwright` is pinned to 1.51.0 to match). **YouTube** is the one
exception: it uses `yt-dlp` because it's a media extractor with a search API, not
an HTML page to stealth-fetch. yt-dlp also handles the media download on every
platform (a browser can't extract signed CDN media).

## Usage

```bash
cardiag doctor                       # preflight: what's installed, what to fix
cardiag train --fixtures             # train a model OFFLINE in ~1.5s (no scrape)
cardiag diagnose clip.wav            # full model: verdict + knock + ranked causes
cardiag triage   clip.wav            # calibrated engine-vs-running-gear
cardiag clean    clip.wav            # isolate the mechanical sound (no model needed)
cardiag inspect  clip.wav -o r.html  # SEE/HEAR the pipeline: spans, spectrograms, scores
cardiag gallery  -o gallery.html     # audio grid of your corpus, grouped by sound-type
cardiag serve                        # local drag-and-drop web app at :8000
cardiag scrape   youtube|tiktok|reddit
cardiag train
```

Add `--json` to any inference command for machine-readable output. New to the
project? Run `cardiag doctor`, then `cardiag train --fixtures && cardiag inspect
<clip>` to see the whole thing without scraping — and read
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

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

## Security & privacy notes

This is a local research tool; a few things are worth knowing before you run it:

- **Model files are trusted input.** `cardiag diagnose --model X.joblib` (and the
  `CARDIAG_MODEL` env var) load a pickle via joblib — loading a model executes code
  in it. Only load `.joblib` models you trained or trust.
- **Reddit scraping reads your browser cookies.** The Reddit audio step shells out
  to `yt-dlp --cookies-from-browser firefox` to fetch `v.redd.it` media, which
  decrypts your logged-in session. Set `CARDIAG_COOKIES_BROWSER` (e.g. `chrome`) or
  skip Reddit if you'd rather it not touch your profile.
- **The web app has no auth.** `cardiag serve` binds `127.0.0.1` by default. Don't
  pass `--host 0.0.0.0` on an untrusted network — it would expose an unauthenticated
  upload/inference endpoint (it warns you if you try).

## Acknowledgements

The pipeline and corpus methodology were developed in two predecessor research
projects; `cardiag` is the cleaned, focused, open-source extraction of the sound
engine. Built on CLAP (`laion/clap-htsat-unfused`), Silero VAD, librosa, yt-dlp,
Camoufox, and scikit-learn.

## License

MIT — see [LICENSE](LICENSE).
