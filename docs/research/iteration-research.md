# Iteration research — tuning the audio→fault model

> Research notes from an iterative tuning run: propose a change, measure it with
> the grouped-CV scorecard, keep only what beats the significance gate. See
> [eval-methodology.md](eval-methodology.md) for the evaluation protocol.

## ⭐ Headline finding (overturns the project's core assumption)

The pipeline's two worst failures — cross-domain cause stuck at baseline, and
**0% knock recall** — are **weak-label problems, not feature-space problems.**
The same frozen CLAP embeddings that score 0% knock zero-shot reach **90% knock
recall** and **86% cause** when a linear head is trained on a few hundred
*clean* labels:

| Regime (same CLAP features) | cause top-1 | fault-vs-normal | knock recall |
| --- | --- | --- | --- |
| trained on **weak scrape** labels | 0.44 | 0.55 (≈chance) | **0.00** |
| trained on **clean verified** labels | **0.86** | **0.88** | **0.90** |

Clean-label numbers are **5-fold grouped-CV means** (not one lucky split):
cause 0.856±0.019, fault-vs-normal 0.872±0.039, knock 0.908±0.027. The knock
result is **leakage-clean** (Car-Engine vehicle-grouped, 202 vehicles / 203
clips → genuine cross-vehicle generalization). *Caveat:* car-diagnostics has no
recording IDs, so its cause ceiling could be mildly optimistic if clips are
segments of shared recordings; the knock ceiling has no such caveat.

**Consequences for the project plan:**

1. **The bottleneck is label quality, not the backbone.** The early premise
   ("need BEATs ≫ CLAP") is *not* what's blocking us — CLAP is already enough
   for these tasks given clean labels (later confirmed: a frozen PANNs/CNN14
   fusion gave no significant gain — see [eval-methodology.md](eval-methodology.md)).
   BEATs is now an optimization, not the
   critical path.
2. **The verified external data should move from eval-only to training.** It is
   the highest-leverage asset in the repo. ~1,900 clean clips beat ~5,000 weak
   ones decisively. (Needs a fresh held-out — split verified by source/vehicle.)
3. **The weak scrape is best used via a clean teacher**: train on verified,
   pseudo-relabel the 12k scrape, keep only high-agreement clips. The scrape
   becomes volume *after* a clean model filters it, not raw training data.
4. The fusion/tiering pipeline **overstamps** — even "gold" weak labels cap the
   head far below the clean ceiling. Trust verified > gold > silver, and treat
   all scrape tiers as candidates to be re-verified, not ground truth.

This is the kind of result the measurement discipline exists to surface: it is
counter to the documented plan, and it is backed by a held-out number with a
leakage-controlled check, not a hunch.

---

## The mapping (what "self", "child", "verifier" are here)

| Self-tuning concept | This project |
| --- | --- |
| Mutable program `self(N)` | the **modeling recipe**: `{backbone, normalization, head, features, label/tier policy}` |
| `child(N)` | one deterministic train+eval run with that recipe (`training1/iterate.py`, fresh process) |
| Verifier (§3.2, must be cheap + trustworthy) | the **verified-external eval** — car-diagnostics / Car-Engine / Ai_Mechanic, real human labels the model never trains on — with **bootstrap 95% CIs** so "better" is a measured interval, not a vibe |
| `observe → diagnose` | which metric failed + *why* (feature space? domain shift? label noise?) |
| `edit` (must generalize, §6.4) | change **one** recipe knob, motivated by a principle |
| Measurement (§5) | predict the metric, measure with CI, check if the gain clears the CI overlap |
| Termination (§7.4) | iteration/fuel cap, or fixed point (nothing beats current outside CI) |

I (one agent, no sub-agents per repo policy) am the runner: I read each child's
metrics, reflect, edit the recipe, re-run. This is orchestration-style tuning
(§2.3.2) where the "worker program" is the recipe.

### The three metrics being raised (all on held-out / verified data)

1. **cause** top-1 / top-3 on verified external (5-class overlap; majority 0.293)
2. **kind** fault-vs-normal acc on verified external (zero-shot rule = 0.712)
3. **L1** sound-type acc + **knock recall** on verified Car-Engine (knock n=87)

In-domain test (creator-split) is reported too, but the *gap* between it and
verified-external is itself a signal (domain shift).

---

## I0 — baseline (CLAP embeddings + L2-norm + logistic regression)

Recipe: `emb=clap, norm=l2, head=logreg, tiers=gold+silver`. The committed
first model, re-measured with CIs.

| Metric | value | 95% CI | reference |
| --- | --- | --- | --- |
| cause ext top-1 | 0.305 | [0.261, 0.348] | majority 0.293 |
| cause ext top-3 | 0.561 | [0.516, 0.607] | — |
| cause **in-domain** top-1 | 0.329 | [0.292, 0.367] | majority 0.189 |
| kind ext fault-vs-normal | 0.642 | [0.617, 0.668] | zero-shot 0.712 |
| L1 ext acc | 0.413 | [0.351, 0.476] | — |
| L1 ext **knock recall** | **0.000** | — | n=87 |

**Diagnosis (the three failure modes to attack):**

- **F1 — cross-domain cause is at baseline.** Top-1 CI [0.261, 0.348] *contains*
  the 0.293 majority rate: in-domain the head learns (top-1 0.329 ≫ 0.189
  baseline), but that signal doesn't transfer. → **domain shift** between
  scrape audio and clean verified recordings. Knob to try: normalization /
  domain-mean-centering; retrieval head; verified-supervised anchors.
- **F2 — trained kind head is *worse* than zero-shot** (0.642 < 0.712). The
  unbalanced logreg over-predicts the train-majority class (fault) on a ~50/50
  verified set. → balanced weighting.
- **F3 — knock recall is exactly 0** despite 87 verified knocking engines, and
  in-domain L1 top-1 is 0.879. The model perfectly reproduces CLAP's blind
  spot. → CLAP **feature space** cannot represent engine knock; needs a
  sound-event backbone (AST/BEATs) and/or cyclic DSP features.

---

## Iteration log

Each row: one knob change, predicted effect, measured (CI), generalized lesson.
Filled in as the loop runs.

| N | recipe Δ (vs current best) | target | predicted | measured (95% CI) | verdict | lesson (generalized) |
| --- | --- | --- | --- | --- | --- | --- |
| I0 | baseline | — | — | see table above | ref | — |
| I1 | head → balanced | kind | beat zero-shot 0.712 | ext fvn **0.744** [0.72, 0.766] | ✅ **WIN** | On a balanced real-world eval, balance the head; an unbalanced head trained on a skewed corpus underperforms even zero-shot. |
| I2 | norm L2 → zscore / dmc | cause | close domain gap | ext top1 0.243 / 0.189 (both ↓) | ❌ revert | Respect the backbone's native geometry (cosine/L2 for CLAP). Per-dim standardization and domain-mean-centering destroy it. Domain gap is **not** first-order-removable. |
| I3 | head logreg → kNN (k=15/50) | cause | retrieval transfers better | ext top1 0.268 / 0.257 (↓) | ❌ revert | Parametric logreg beats raw kNN cross-domain — kNN is hostage to scrape density. (Retrieval may still serve the app's "nearest reference clip" UX, but not as the classifier.) |
| I4 | + DSP cyclic/spectral features (z-scored, scale 0.3) | cause, L1 | cyclic structure lifts knock | cause 0.27 (↓), knock **0.0** | ❌ no effect | Concatenating hand DSP features onto CLAP doesn't fix knock — because knock isn't missing from the features (see ceiling), it's missing from the *labels*. Right fix, wrong layer. |
| **V** | **train on clean verified labels** (½ split, vehicle-safe) | cause, kind, knock | clean labels help | **cause 0.86, kind 0.88, knock 0.90** | ✅ **DECISIVE** | Label quality, not feature space, is the ceiling. Verified data → training is the highest-leverage change. See headline. |
| I5 | backbone CLAP → **AST** (AudioSet) | all | sound-event backbone beats CLAP | ceiling cause 0.84 / kind 0.85; held-out fvn 0.874 / knock 0.92 / cause 0.86 | ❌ ≈tie | A different, AudioSet-pretrained backbone does **not** beat CLAP — confirms the backbone is not the bottleneck. The documented "need BEATs ≫ CLAP" premise is empirically wrong *for this data*. |
| E2 | labels weak-fusion → **text-mined** (same clips) | cause | better labels lift verified | 0.268→0.295 top1, 0.51→0.54 top3 (CIs overlap) | ◐ small | Full-text re-labeling helps a little but not significantly on the coarse verified eval (labels agree 88%). Its real payoff is specificity + new capability (E3), not a bump on 5 coarse classes. |
| E3-A | text-mined region labels | engine vs chassis | audio localizes the fault | 0.77 acc / **0.73 macro-recall** (maj 0.73) | ✅ usable | Audio separates engine-internal from chassis clunks — clinically useful triage. |
| E3-C | text-mined subtype labels | **rod knock vs lifter** | audio tells knock subtypes apart | **0.69 acc / 0.69 macro-recall** (maj 0.58, ±0.17) | ✅ preliminary | Text-mining made knock-subtyping *answerable* (0 → 114 labels); audio partially distinguishes rod-bearing knock from lifter tick. Wide CI (27 videos) — needs more data + human verification. |

### Reflections after the CLAP-only round (I1–I3)

Two of three failure modes resist *any* CLAP-side fix: F1 (domain gap) survived
normalization and retrieval; F3 (knock) is untouched because it's a property of
the **feature space**, not the head or the labels. The CLAP-only knobs are
exhausted — only the head-balancing win (I1) survives. The remaining attacks
must change the **features**: add DSP cyclic structure (targets F3 directly) or
swap to a sound-event backbone (AST/BEATs, targets F1+F3). Those caches are
building; iterations continue below.

**Current best recipe:** `clap, l2, logreg` for cause/L1; `logreg-balanced` for
kind. (I2/I3 reverted.)

---

## Converged recipe & recommended next phase

The loop reached a fixed point on the *modeling* knobs (nothing beat the I0
geometry except head-balancing) but in doing so surfaced that the real lever is
**data**, not model. The converged recommendation:

### Deployable model today (CLAP + linear heads)

- **fault-vs-normal**: CLAP L2 + logistic-**balanced**. Verified acc 0.74
  deployed (scrape-trained); **0.87 achievable** once trained on verified data.
- **L1 sound-type**: CLAP L2 + logistic. Strong in-domain (0.88); for knock
  specifically, train on the verified Car-Engine clips (0% → 0.90).
- **L2 cause**: CLAP L2 + logistic, **top-3 + confidence + nearest-clip**, never
  a single answer (top-1 cross-domain is weak; top-3 ~0.56 weak-trained,
  ~0.99 clean-trained).

### Deployable model, built & held-out-evaluated (`train_best.py`)

The recommendation, executed: clean teachers trained on 75% of verified groups,
tested on a **leakage-safe** 25% held-out (group-disjoint by vehicle/source).
These are the honest numbers a v1 app would ship with — and they're produced by
a script that exists now, not a plan:

| Head | held-out acc | 95% CI | majority | note |
| --- | --- | --- | --- | --- |
| fault-vs-normal | 0.848 | [0.805, 0.887] | 0.649 | all verified, vehicle/source-grouped |
| knock-vs-normal | 0.952 | [0.889, 1.000] | 0.540 | Car-Engine, vehicle-grouped, **knock recall 1.0** |
| cause (6-class) | 0.873 | [0.813, 0.925] | 0.306 | car-diagnostics |

Saved to `training1/data/best_model_clap.joblib`. Compare: the deployed
scrape-trained model scored 0.74 / 0.00 / 0.31 on the same kinds of tasks.

**The app exists too** (`predict.py`): audio file → fault verdict + knock
warning + top-3 cause with confidence. On a verified Toyota knocking-engine clip
it correctly raises *ENGINE KNOCK p=0.72* and surfaces low-oil as the top cause
— the exact clip family the original zero-shot pipeline scored 0% on.

### The three changes that matter (in priority order)

1. **Promote verified data to training.** Build `training2/` that splits the
   ~1,900 verified clips into grouped train/test (by vehicle/source), trains on
   the verified-train + held-out verified-test. This single change is worth more
   than any backbone swap (44%→86% cause, 0%→90% knock).
2. **Clean-teacher pseudo-labeling of the scrape.** Train the clean teacher,
   relabel all 12k scrape clips, keep only where teacher⋅weak-label agree (and
   confidence high). Naively mixing scrape into clean training *diluted* it
   (hybrid 0.80 < verified 0.86) — so filter first, then mix.
3. **Stop trusting tiers as truth.** "gold" weak labels still cap the head far
   below the clean ceiling; the fusion stage overstamps. Tiers are a *triage
   prior*, not ground truth — re-verify before training on them.

### Deprioritized by this research

- **BEATs/AST backbone swap** — was the documented #1 next step; demoted to an
  optimization. (AST measured below for completeness; the ceiling is already
  high with CLAP, so a better backbone can only raise an already-good number,
  not unblock a stuck one.)
- **Scaling the scrape 3–5×** — more *weak* labels is low-value until the
  clean-teacher filter exists; otherwise it's more noise.

### Open threads / next iterations

- ~~AST ceiling vs CLAP~~ — done (I5): ≈tie, backbone is not the lever.
- Multi-window / TTA averaging per clip.
- `l1_conf` recalibration (isotonic) before any confidence-weighted training.
- A truly held-out verified test that no tuning ever touches (current verified
  set is doing double duty as both verifier and ceiling-prober).

---

## Can we diagnose the *specific* cause of engine knock? (+ text-mining the fix)

Short answer: **not from audio alone with today's labels** — and the reasons are
the same label-quality story.

### Why knock-subtyping is hard (data-grounded)

1. **The sound is one-to-many.** Of 714 verified-ish "knock/clunk" clips, only
   ~12% are engine-internal; the rest are *suspension/driveline clunks* (ball
   joint, CV joint, sway bar, mounts, struts) that sound nearly identical to a
   mic. The L1 "knock" class conflates engine knock with suspension clunk.
2. **No clean labels for knock subtypes exist in the corpus.** Verified
   Car-Engine is a flat "knocking" bucket (87/100 just "knock"); car-diagnostics
   has `low_oil` but not rod/lifter/detonation; DB1 has ~1 clip each of "thrown
   rod" / "lifter ticking". Nothing trainable or evaluable.
3. **It's partly non-acoustic** (rod vs lifter vs detonation needs RPM/load,
   cold-vs-warm, stethoscope localization, oil check) — bounded by the audio
   ceiling regardless of labels.

So the app should treat knock as **triage** ("engine-internal or a suspension
clunk; localize + check oil"), not a confident subtype call.

### The fix being prototyped: mine the text we already have (`text_mine.py`)

The corpus holds **1,116 full transcripts, 256 chaptered videos, descriptions
to ~5k chars, 18 comments/video** — but `fusion.py` only saw a per-clip
transcript *slice* + one comment. A mechanic saying *"that's a rod knock,
you're low on oil"* elsewhere in the video was invisible to it. Haiku reading
the **whole** video text recovers specific, timestamped, explicitly-stated
subtypes the per-clip fusion missed. Pilot (8 knock videos) yielded e.g.:

- `rod bearing` [stated], times [105-119, 184-200, 263-330]
- `crankshaft bearing` [132-144], then `clutch`, then `belt` — one "guess the
  noise" video, three faults at three timestamps
- `intake phaser` cold-start rattle (named Toyota VVT fault)

**Why this is the right lever, not over-trusting an LLM** (the overstamp lesson,
§6.6): each label carries an `explicit` flag (a human actually said it) and
`time_ranges`, so labels join to clips *by timestamp* and we trust
explicit+timestamped ≫ inferred, and cross-check against audio. Haiku generates
candidates; agreement makes them trustworthy.

### Results — full corpus mined (507 videos, `join_sublabels.py`)

**1,860 clips sub-labeled, 1,054 high-trust** (timestamp + explicitly stated),
1,841/1,860 explicit. The engine-knock subtypes now have trainable counts:
rod bearing 63, lifter 73, rocker arm 51, plus wheel bearing 142, water pump 23,
belt 31, and long-tail specifics the 24-group vocab can't express ("stone stuck
between brake disc and backplate" ×16, "torque converter bearing", "rotor rust").

**E2 — does better labeling beat weak fusion?** Same 701 clips, same CLAP
features, two cause heads differing *only* in labels (weak fused_cause vs
text-mined), both eval'd on verified external:

| labels | verified top-1 | top-3 |
| --- | --- | --- |
| weak fusion | 0.268 [0.23, 0.31] | 0.509 |
| text-mined | 0.295 [0.255, 0.34] | 0.543 |

A real but **modest, not-significant** lift (CIs overlap). The two label sets
agree 88%, so the marginal effect is bounded, and the coarse 5-class verified
eval can't see text-mining's real win — *specificity* and *new capability*
below. (The dominant error on this eval is the scrape→verified domain gap, not
label noise within the overlap classes.)

**E3 — what can audio predict from the sub-labels?** (video-grouped CV,
leakage-safe; labels text-derived, high-trust but not human-verified):

| rung | task | n / videos | result | majority |
| --- | --- | --- | --- | --- |
| A region | engine-internal vs chassis clunk | 493 / 128 | **0.77 acc, 0.73 macro-recall** | 0.73 |
| B part | 8-way part (better labels) | 534 / 140 | 0.31 top-1 / **0.57 top-3** | 0.24 |
| C **knock subtype** | **rod bearing vs lifter** | 114 / 27 | **0.69 acc, 0.69 macro-recall** | 0.58 |

**The answer to "can we diagnose the specific cause of engine knock":** before
text-mining, *unanswerable* (0 trainable subtype labels). Now, a qualified
**yes** — audio distinguishes rod-bearing knock from lifter tick at ~69%
(±0.17; only 27 videos, so wide), and localizes engine-vs-chassis at ~73%
balanced recall. Both beat chance/majority. Neither is product-ready, but the
direction is real and the path is concrete: more mined videos + human
verification of the high-trust subtype clips → firm up the CI → ship as triage.
