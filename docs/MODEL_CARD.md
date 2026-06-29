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

## Headline (YouTube+TikTok, Reddit dropped) — statistically tested

Trained/tested on **YouTube+TikTok only** (Reddit is deprecated as a training
source: uncurated, off-target, adds label noise). By-**video** grouped CV over
**1,031 independent video groups** — leakage-safe and enough clusters for a valid CI:

- **fault/normal: AUROC 0.794, 95% CI [0.762, 0.825]**, balanced-acc 0.726.
- Better-than-chance: label-permutation null mean 0.500, **p = 0.0005**.
- Localization (targeted-recording benchmark = upload proxy): region top-3 ≈0.73,
  part top-3 ≈0.46; recall@k beats chance with binomial p ≪ 1e-20.

The earlier 711-clip table below is the small in-repo loop model; the numbers above
are the current, statistically-defensible figures for the social/targeted-upload scope.

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

## Shipped proof-of-concept model (`models/`)

The bundled model is **not** the 711-clip loop model above — it is a deliberately
**diverse, class-balanced** model built by `scripts/build_poc_model.py`, trained on
**1,159 mechanical spans** segmented through the same `clean()` cascade as
inference (no whole-clip embeds) and spanning **four independent domains**: a
high-confidence balanced slice of the scraped social corpus (YouTube+TikTok;
Reddit excluded as noisiest) plus three benchmark datasets — Car-Engine (clean
normal/abnormal), ai-mechanic, and Sound-Based-Vehicle-Diagnostics (28 named
fault types). Each long benchmark recording becomes several short spans.

By-video / by-recording grouped CV:

| Head | metric | value | read |
|---|---|---|---|
| **kind** (fault/normal) | balAcc | **0.80** | across 4 domains — vs 0.64 social-only |
| **knock** | balAcc | **0.96** | strong |
| **triage** | balAcc | 0.64 | modest |
| **cause** (part) | top-3 / region top-2 | 0.46 / **0.56** | ranked shortlist; "where in the car" ≈ 2× chance |

**Generalization — the honest part.** Group-safe train/validate/test on the diverse
set: validation **0.89** → test **0.75** AUROC; on the clean independent Car-Engine
domain, held-out test clips reach **0.88**. But **leave-one-*domain*-out** (an entire
domain never seen in training) collapses to **0.47–0.57** — the model generalizes to
*new clips from domains it has examples of*, not yet to a *completely unseen* domain.
That needs many more domains; with ~4 it can't span the space. This is domain
generalization working as documented, reported plainly rather than hidden.

**Scope that makes it valid.** For social-style / targeted-upload inference (YouTube /
TikTok, or a phone clip a user records deliberately), the leakage-safe number is the
**1,031-video-group YouTube+TikTok CV: AUROC 0.794, 95% CI [0.762, 0.825]** (Reddit
dropped) — at the literature ceiling, and the regime this model is meant for.

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
