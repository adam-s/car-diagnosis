# Model card — cardiag

Following Mitchell et al., *Model Cards for Model Reporting* (FAT\* 2019). All
numbers are leakage-safe (by-video grouped CV) and reported out-of-sample where
possible. The honest case for why a deliberately crude method earns these numbers
is in [DEFENSE.md](DEFENSE.md).

## Model details

- **What:** small scikit-learn linear heads (`LogisticRegression`, class-balanced,
  temperature-calibrated) on top of a **frozen** CLAP audio encoder
  (`laion/clap-htsat-unfused`, 512-d). Heads:
  - `kind` — fault vs normal,
  - `region` — *where in the car* (6 zones: engine / accessory / exhaust /
    drivetrain / suspension-steering / brakes-wheels) — the headline localizer,
  - `cause` — finer part family, served as a ranked top-3,
  - `triage` — engine-internal vs running-gear,
  - `knock` + `knock_region` — a coarse-to-fine cascade: detect a knock, then a
    specialist localizes it (soft-routed by `P(knock)`).
- **Embedding contract:** every clip — at train and at inference — is cleaned by the
  same cascade (VAD + energy + spectral flatness + CLAP music gate), split into
  ≤10 s windows, and embedded per-span by the same `embed_clip()`; multi-span clips
  are pooled in probability space. No train/serve skew.
- **Shipped model:** a pre-trained model ships in `models/`, built by
  `scripts/build_poc_model.py`. A fresh clone can also train its own with
  `cardiag scrape … && cardiag train`.

## Intended use

- **Intended:** a calibrated **triage aid** — flag fault vs normal, narrow *where in
  the car* the noise is, and offer a ranked shortlist of likely parts. A reusable
  demonstration of an honest scrape→clean→train→serve loop.
- **Out of scope:** a safety-critical or standalone diagnostic. Do not make
  repair/no-repair decisions from it without a mechanic.

## Training data

The shipped model is **diverse and class-balanced**: a high-confidence balanced
slice of the scraped social corpus (**YouTube + TikTok**; Reddit dropped — uncurated
and off-target) plus three independent benchmark datasets (Car-Engine, ai-mechanic,
Sound-Based-Vehicle-Diagnostics), all segmented through the same cleaning cascade
into short mechanical spans. Weak labels: `kind`/`cause` from query set and
title/caption keywords; benchmark folders carry explicit causes. No LLM.

## Evaluation — leakage-safe, out-of-sample

**Headline (YouTube+TikTok, by-video grouped CV over 1,031 independent video
groups — valid CI, no pseudo-replication):**

| Question | Result | vs chance |
|---|---|---|
| fault / normal | **AUROC 0.794, 95% CI [0.762, 0.825]**, balAcc 0.726 | permutation p = 0.0005 |
| "where in the car" (6 zones) | right zone in **top-3 ≈ 0.75** | binomial p ≪ 1e-30 |
| which part (12+ families) | **top-3 ≈ 0.45–0.65** | 3–4× chance |
| engine vs running-gear (triage) | AUROC ≈ **0.79** | — |

These sit in the documented literature band for in-the-wild machine-sound
classification (DCASE Task 2: mid-70s–low-80s AUROC). The same method reaches
**0.93 AUROC** trained+tested in-domain on clean engine audio — so the ceiling here
is the *data*, not the method.

**The knock head, reported honestly.** In-distribution it scores AUROC 0.99; on a
held-out *engine-knock* set it collapses to **0.56 (chance)**. The cause: "knock" is
one acoustic label worn by ~24 different parts (suspension dominates; engine
rod-knock is a minority), and we trained on mostly chassis clunks then tested on
engine knock. It is therefore **demoted** — kept as a hint, never the headline — and
replaced as a localizer by the `knock_region` specialist (cv-balAcc 0.52 vs the
general region head's 0.29).

**Generalization limit.** Leave-one-*domain*-out (an entire recording domain never
seen in training) drops to 0.47–0.57: the model generalizes to new clips from
domains it has examples of, not yet to a completely unseen domain. That needs many
more domains; with ~4 it can't span the space. Reported, not hidden.

## Calibration

Heads are temperature-scaled (Guo et al. 2017) on out-of-fold logits; measured ECE
on `kind` ≈ 0.04. A weak head is pulled toward 0.5, so `diagnose` returns
**UNCERTAIN** rather than a confident wrong verdict.

## Limitations

- Within-social confound: normals are YouTube-heavy. The one untapped data lever is
  scraping balanced normals from TikTok (`cardiag scrape tiktok --normal`).
- Fine `cause` has low-count classes — trust the ranked shortlist, not a single class.
- Architecture changes (whitening, kNN, fusion) showed no significant gain; the
  binding constraint is data, not the head.

## Ethical / safety

A confident-sounding wrong answer could lead to an unnecessary or skipped repair.
Mitigated by calibration (UNCERTAIN when weak), ranked-not-single output, and
explicit "triage, not a final diagnosis" framing. Model files execute code on load
(joblib) — load only models you trust.
