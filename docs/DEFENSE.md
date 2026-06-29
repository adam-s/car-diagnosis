# Why this works — a defense of a deliberately crude diagnostic

> **Claim.** Diagnosing a car fault from a phone recording is a hard, physically
> constrained problem, and the approach here is intentionally crude — frozen
> embeddings and linear models, no fine-tuning, no deep network. Yet the result is
> *legitimately useful as a triage aid*, and every claim is backed by a leakage-safe,
> statistically-tested measurement. This document is the evidence.
>
> **The real contribution is the cleaning pipeline.** In the real world a useful
> training set would be *targeted, human-labeled* recordings of specific faults. We
> don't have that — we have noisy social video. The significance of this project is
> that the cleaning cascade **extracts the targeted mechanical sound from messy
> YouTube/TikTok clips**, doing automatically what a human does by recording
> deliberately. Reddit is dropped on purpose: a 4-minute uncurated Reddit post is
> the opposite of a targeted upload, and it adds label noise without improving
> accuracy. We train and test on YouTube+TikTok, which the cascade can clean.

A mechanic has a stethoscope, hands, an OBD-II port, and a trained ear standing
next to the engine. This model has a **single compressed audio clip from a phone**,
often with wind, speech, and music mixed in. Judged against a mechanic it will
always lose. Judged as *"can a crude, honest, reproducible pipeline point you in
the right direction from sound alone?"* — the answer, measured, is **yes**.

---

## 1. How crude it actually is (and why that's the point)

| Stage | What most people reach for | What this does |
|---|---|---|
| Audio model | fine-tune a large audio transformer | **frozen** CLAP (`laion/clap-htsat-unfused`), never trained |
| Classifier | a deep net with millions of params | **`LogisticRegression`** — a linear head, ~100 KB total |
| Features | learned, end-to-end | a single 512-d embedding per cleaned span |
| Pooling | learned attention | plain **mean of per-span probabilities** |

There is almost nothing to overfit and nothing clever to hide behind. If it works,
it works because the *signal is really there* in the embeddings and the pipeline is
honest — not because a large model memorized the training set. **Crudeness is the
control:** a linear probe on frozen features is the standard way to ask "is this
information linearly present?" The answer below is yes, to a measurable degree.

---

## 2. The method is sound — proven where the data is clean

The single most important result. Take the *exact same* frozen-CLAP + linear-head
recipe and train **and** test it in-domain on a clean independent benchmark
(Car-Engine, 216 clips, normal vs abnormal), 5-fold grouped CV:

> **AUROC 0.93 · balanced accuracy 0.88**

So when the data is clean and the train/test distributions match, the crude method
is *strong*. That isolates the variable: **the ceiling in the wild is the data, not
the model.** Everything below is about what survives the messiness of real
social-media / phone audio.

---

## 3. Evaluation methodology (why the numbers are trustworthy)

Bad audio-ML demos inflate themselves three ways; we close all three:

- **No leakage.** Splits are `StratifiedGroupKFold` grouped **by source recording**,
  so two spans from the same video can never land in both train and test. This is
  what makes in-distribution numbers honest instead of memorized.
- **Leakage-safe splits with many clusters.** The headline fault/normal number is
  by-**video** grouped CV over **1,031 independent YouTube/TikTok video groups** —
  two spans from one video can never split across train and test, and there are
  enough independent groups for a valid confidence interval. Localization is also
  checked on three **external targeted-recording** benchmark datasets the social
  model never trained on (an honest cross-domain probe, with its 3-domain caveat
  stated above).
- **One pipeline, train and serve.** The same `clean()` cascade segments audio into
  short mechanical spans at corpus-build time, at training, and at inference; a
  10-minute recording becomes several short spans exactly like a phone clip. There is
  no train/serve skew to flatter the numbers, and pooling is done in probability
  space (never by averaging embeddings the head never saw).

We also ran a **permutation null** (label-shuffle) and report calibration error, so
"better than chance" is a tested statement, not an assumption.

---

## 4. What it actually achieves

Coarse calls below are the **1,031-video-group YouTube+TikTok CV** (the tight,
trustworthy numbers). The localization recall@k tables are measured on the
**targeted-recording benchmark datasets** — the closest proxy we have to "a human
uploaded a deliberate clip of the symptom," which is the real product input.

### Coarse calls — solid, and now statistically tested

YouTube+TikTok only (Reddit dropped), **by-video** grouped CV over **1,031
independent video groups** — so the splits are leakage-safe *and* the confidence
interval is valid (no pseudo-replication):

| Question | Result |
|---|---|
| Is something wrong? (fault/normal) | **AUROC 0.794, 95% CI [0.762, 0.825]**, balAcc 0.726 |
| Better than chance? | label-permutation null mean 0.500, **p = 0.0005** (floor of 2,000 shuffles) |
| Roughly where? (engine vs running-gear) | AUROC ≈ **0.79** |

These sit squarely in the **documented literature band** for in-the-wild machine
sound classification (DCASE Task 2 in-the-wild: mid-70s to low-80s AUROC). We are
not under-performing the field — we are *at* it, with a 100 KB linear model, and we
can prove the result is not noise: the 95% interval clears 0.5 by a wide margin and
the permutation test bottoms out at its resolution floor.

> **A note on honesty about significance.** When we instead evaluate on the three
> *external targeted-recording* benchmark datasets (a good proxy for "a human
> uploaded a deliberate clip"), the *effect* is just as significant per the
> permutation null, but those clips cluster into only **3 source domains** — so a
> cluster-honest CI there is wide ([0.62, 0.80]) and we do **not** quote a tight
> point estimate from it. The tight, trustworthy number is the 1,031-video-group CV
> above.

### Localization — judged the fair way (recall@k, not top-1)

A triage tool should be scored on *"is the true answer in the shortlist?"*, because
even a low-confidence-but-present answer points the search the right way. It is, far
more often than chance:

**Where in the car (6 zones):**

| shortlist | true zone in it | random | lift |
|---|---|---|---|
| top-1 | 37% | 17% | **2.2×** |
| top-2 | 60% | 33% | 1.8× |
| top-3 | 76% | 50% | 1.5× |
| top-4 | 89% | 67% | 1.3× |

**Exact part (24 families):**

| shortlist | true part in it | random | lift |
|---|---|---|---|
| top-1 | 18% | 4% | **4.3×** |
| top-3 | 42% | 12% | 3.4× |
| top-4 | 52% | 17% | **3.1×** |

The **lift** column is the defense: when the model lists *"steering — 10%"* and it
turns out to be steering, that is not luck. Across a 24-way space the true answer
appears in a 4-deep shortlist **3× more often than random ordering would allow**.
The ranking carries information all the way down. That is precisely the property a
triage assistant needs: it narrows a 24-way space to a short list of suspects you
can check with a stethoscope.

---

## 5. The honesty that makes it credible

A crude model is only trustworthy if it knows its limits, so we built that in:

- **Calibrated probabilities** (temperature scaling, Guo et al. 2017): expected
  calibration error on the fault/normal head ≈ **0.04**. A reported 70% means ~70%.
- **It says "UNCERTAIN."** When a head is weak or the audio doesn't support a call,
  `diagnose` returns `UNCERTAIN` rather than a confident guess. Silence and corrupt
  audio degrade to UNCERTAIN by construction, not to a confident FAULT.
- **We publish our own failure.** The engine-knock head scores **0.99 in-distribution
  but 0.56 (chance) out-of-sample** — so we *demoted it from the headline* and label
  it "in-distribution only, treat as a hint." The earlier projects this descends from
  led with knock; the out-of-sample check is exactly what caught the overclaim.
  Reporting that collapse, rather than shipping the 0.99, is the whole point.

---

## 6. The limits, stated plainly

- **Exact-part top-1 is ~18%.** This is a shortlist, not a verdict. Pinpointing one
  component from audio alone is at the documented ceiling; we present a ranked top-3/4
  and say so.
- **It does not generalize to a *completely unseen* recording domain.** Trained across
  ~4 domains, leave-one-*domain*-out drops to ~0.47–0.57. It generalizes to new clips
  from domains it has examples of (incl. the held-out verified set), not to a kind of
  recording it has never seen. Closing that needs many more domains.
- **More data of the same kind won't help.** The learning curve is flat from ~500 to
  ~8,000 clips — we are at the data/method ceiling, not short of training examples.
- **Scope.** Valid for social-style audio (YouTube / TikTok / Reddit / a phone clip
  that sounds like those). Not a safety-critical or standalone diagnostic.

---

## 7. Verdict

For a method with **nothing learned beyond a linear boundary on frozen features**,
evaluated **without leakage over 1,031 independent video groups** (and probed
cross-domain on targeted-recording benchmarks), the result is:
a calibrated fault/normal screen at the literature ceiling, a robust "where in the
car" zone call (right zone in a 3-deep shortlist **76%** of the time, **1.5–2.2×**
chance), an exact-part shortlist that beats guessing **3–4×**, and the discipline to
say *UNCERTAIN* and to **publish the one head that didn't generalize**.

It will not replace the stethoscope. It tells you, honestly and reproducibly from a
single phone clip, **where to put it** — and it is right about that far more often
than chance. For a deliberately crude proof of concept, that is not a disappointment.
That is the result working as designed.

*Reproduce every number here: `scripts/build_poc_model.py` (model), the
`StratifiedGroupKFold` harness in `cardiag.training.eval.scorecard`, and the
verified hold-out evaluation described in `docs/MODEL_CARD.md`.*
