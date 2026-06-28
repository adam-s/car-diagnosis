# Model card — cardiag clean-teacher heads

Following Mitchell et al., *Model Cards for Model Reporting* (FAT\* 2019). Numbers
are measured by the committed scorecard
([`python -m cardiag.training.eval.scorecard`](../src/cardiag/training/eval/scorecard.py));
see [SCORECARD.md](SCORECARD.md) and the methodology in
[research/eval-methodology.md](research/eval-methodology.md).

## Model details
- **What:** four scikit-learn linear heads (`LogisticRegression`, class-balanced,
  temperature-calibrated) on top of a **frozen** CLAP audio encoder
  (`laion/clap-htsat-unfused`, 512-d). Heads: `kind` (fault/normal), `knock`
  (engine knock vs normal idle), `cause` (12-way part family, served as a ranked
  top-3), and a separate `triage` head (engine-internal vs running-gear).
- **Embedding contract:** every clip — at train and at inference — is cleaned by
  the same cascade (VAD + energy + spectral flatness + CLAP music gate) and
  embedded per-span by the same `embed_clip()`; multi-span clips are pooled in
  probability space. No train/serve skew.
- **Version:** trained from this repo's scraped corpus; reproduce with
  `cardiag scrape … && cardiag train`. Not a fixed checkpoint — the repo ships
  *no* bundled model (clone-and-train by design).

## Intended use
- **Intended:** an educational, reproducible demonstration of the full
  scrape→clean→label→train→serve loop; a triage aid that narrows a noise to
  engine-internal vs running-gear and offers a ranked shortlist of likely parts.
- **Out of scope:** a safety-critical or standalone diagnostic. Only the `knock`
  head is strong enough to act on alone. Do not use to make repair/no-repair
  decisions without a mechanic.

## Training data
- **711 clips / 120 videos**, scraped from YouTube (438), TikTok (211), Reddit
  (62). Weak labels: `kind` from the source query set; `cause` from video-title
  keyword matching; no human annotation, no external datasets.
- **Known bias (important):** every `normal` clip is from YouTube; TikTok and
  Reddit are 100% fault. Source is therefore partly predictable from audio
  (balAcc ≈ 0.65), so the all-sources fault/normal number is confound-inflated —
  we report the **YouTube-only** figure as honest.

## Evaluation — by-video StratifiedGroupKFold, 5×5 repeats (no leakage)

| Head | metric | value | baseline | read |
|---|---|---|---|---|
| **knock** | AUROC / balAcc | **0.99 / 0.94** | 0.5 | strong, act-on-able |
| **kind** (YouTube-only) | AUROC / balAcc | 0.70 / 0.63 | 0.50 | real (perm p=0.035), modest |
| **triage** | AUROC / balAcc | 0.72 / 0.66 | 0.50 | real, modest |
| **cause** | **top-3** (shortlist) | **0.69** | 0.25 | useful ranked shortlist |
| cause | top-1 | 0.40 | 0.08 | a single answer is *not* reliable — by design |

Literature ceiling for in-the-wild machine-sound classification is mid-70s–low-80s
AUROC (DCASE Task 2); kind/triage sit in that band, knock exceeds it, cause's
ranked-shortlist regime is the documented norm for fine-grained audio fault ID.

## Calibration
Heads are temperature-scaled (Guo et al. 2017) on out-of-fold logits. Measured ECE:
`kind` 0.317 → 0.044, `triage` 0.199 → 0.131. A weak head (kind T≈8) is pulled
toward 0.5, so `diagnose` honestly returns UNCERTAIN rather than a confident
wrong verdict.

## Limitations
- Source confound caps the trustworthiness of `kind` until normal clips are
  scraped from TikTok/Reddit.
- `cause` has classes with 2 examples (`water_pump`, `differential`) — unreliable
  individually; trust the ranked shortlist, not a single low-count class.
- Model-architecture changes (CL2N, PCA-whiten, prototypical kNN, PANNs fusion)
  showed **no significant gain** — the binding constraint is data, not the head.

## Ethical / safety
Misdiagnosis risk: a confident-sounding wrong answer could lead to an unnecessary
or skipped repair. Mitigated by calibration (UNCERTAIN when weak), ranked-not-
single cause output, and explicit "triage, not a final diagnosis" framing in
`diagnose`. Model files execute code on load (joblib) — treat as trusted input.
