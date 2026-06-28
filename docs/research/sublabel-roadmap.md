# Sub-label accuracy — deep review & overnight roadmap

> Companion to [iteration-research.md](iteration-research.md). Written mid-run,
> 2026-06-12. Goal set by Adam: drive sub-type (specific failing part) accuracy
> toward 99% using the text-extracted labels.

## TL;DR — the honest target (read this first)

"99% accuracy on sub-labels" splits into **two numbers that must not be
conflated**, because one is achievable and one is bounded:

1. **Text-label precision** — when a mechanic *explicitly says* "that's a rod
   knock" at the moment the clip plays, that label is ~right. This can be
   **~99% precise**, and we can *measure* it by spot-verifying a sample. This is
   a realistic target and it's the asset the whole project now rests on.
2. **Audio→sub-label accuracy** — a model naming the part from sound *alone*.
   This is bounded by physics (some subtypes are acoustically near-identical;
   some are partly non-acoustic) and by data (video count). **99% accuracy on
   *every* clip from audio alone is not physically achievable** — and saying
   otherwise would violate the measurement discipline this project runs on.

**The achievable, useful reframing: 99% *precision* at controlled *coverage*.**
The model commits to a sub-type only when confident, and on the clips it commits
to, it is ~99% right; otherwise it abstains or returns top-k + nearest reference
clip. This is how real diagnostic triage works, and it's a goal we can actually
hit and measure. Every iteration below is aimed at **raising coverage at fixed
high precision**, not at a mythical accuracy-on-all.

---

## Part 1 — What we've done (the arc, with the numbers)

| Phase | Result |
| --- | --- |
| Cleanup + org | repo under git, junk gone, shared/ + training1/ structure |
| Training manifests | leakage-safe train/val/test, 24-group canonical causes |
| Calibration vs verified | zero-shot CLAP cause 0.46; **knock recall 0%** |
| **Headline finding** | **label quality, not the CLAP feature space, is the ceiling**: same features, weak→clean labels = cause 0.44→0.86, fault/normal 0.55→0.88, **knock 0.00→0.90** (leakage-verified, 202 vehicles) |
| Backbone test (AST) | AST ≈ CLAP — backbone is *not* the lever (BEATs premise wrong here) |
| Deployable model + app | held-out verified: fault/normal 0.85, knock 0.95, cause-6 0.87; `predict.py` flags real engine knock |
| **Text-mining (the pivot)** | Haiku over full transcripts+chapters+desc recovers specific, **timestamped, explicitly-stated** sub-labels the per-clip fusion threw away |
| Sub-labels (current) | **2,471 clips sub-labeled, 1,656 high-trust, 416 videos** (YT timestamp-joined + TikTok) |
| Knock-subtype, first look | rod-bearing vs lifter **0.69** acc; engine-vs-chassis **0.77** — both > chance; *answerable for the first time* |

The decisive realization: the bottleneck was never the audio model. It was the
labels. Text-mining is how we manufacture clean labels at scale from text we
already paid to collect.

## Part 2 — What we're doing right now (in flight tonight)

1. **TikTok mining** (`text_mine_tt.py`) — 797 fault videos via query+desc+OCR.
   Already added 611 sub-labels / lifted video count 140→416. *More videos is
   the single most important lever* (it's what bounds the CV confidence).
2. **Physics DSP features** (`dsp_features2.py`) — envelope-spectrum modulation
   rate (rod ~crank rate vs lifter ~cam rate), band-energy ratios, impulsiveness
   (crest factor, envelope kurtosis), onset-flux, MFCC timbre. With pre-emphasis
   to boost the high-frequency content where tick/lifter energy lives. Concat
   with CLAP, z-scored. **This is the answer to "emphasize higher-pitch noises"**
   — grounded below.
3. **Tightness signal** (`tm_range_dur`) — prefer labels where the host cued
   *this* sound (short overlap) over whole-chapter spans. A label-quality knob.
4. **Augmentation + MLP harness** (`sublabel_experiment.py`) — mixup + noise in
   feature space for the small-data classes; MLP head option.

## Part 3 — Audio preprocessing research (answering "sound manipulation to raise fidelity")

Yes — there's a deep literature, and the techniques most relevant to *fault
sounds* (transient, rotating-machinery) are specific:

- **Pre-emphasis / high-frequency emphasis** — a first-order high-pass that
  boosts highs (exactly your "increase volume of higher-pitch noises"). Standard
  in speech; helps where the discriminative energy (lifter tick, hiss) is
  high-band. *Applied in `dsp_features2.py`.*
- **PCEN (Per-Channel Energy Normalization)** — replaces log-mel; an adaptive
  AGC per frequency band that **enhances transients** and suppresses stationary
  drone. Shown to beat log-mel for transient event detection. *Strong candidate
  for a re-embedding pass.*
- **HPSS (Harmonic-Percussive Source Separation)** — splits pitched/tonal
  (whine, hum — *harmonic*) from impulsive (knock, tick — *percussive*). Knock
  and tick are percussive "vertical" spectrogram structures; isolating the
  percussive component should sharpen them. *(Tried as a feature; too slow at
  scale — revisit as a targeted re-embed of just engine clips.)*
- **Envelope-spectrum / order analysis** — *the* classic for bearing/rotating
  faults: rectify→envelope→FFT exposes the periodic impulse rate (bearing
  characteristic frequencies; rod knock at crank order, lifter at cam order =
  half). Physically the most discriminative cue for our hardest split. *Core of
  `dsp_features2.py`.*
- **CQT / log-frequency** vs mel — better low-frequency pitch resolution (rod
  knock is low-frequency).
- **Augmentation chains** (SOTA, DCASE): SpecAugment (time/freq masking) +
  mixup + pitch/time perturbation + added noise at varied SNR, applied as a
  *chain*. The fix for small per-class video counts.
- **Test-time augmentation** — average predictions over multiple crops/pitch
  shifts (light version already in `predict.py`).

Sources: [Sound classification augmentation review (MDPI)](https://www.mdpi.com/2079-9292/11/22/3795) ·
[DCASE augmentation chains](https://arxiv.org/pdf/2209.01802) ·
[Envelope spectrum for bearing faults (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11085194/) ·
[Sub-band envelope power spectrum (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5981466/) ·
[HPSS tutorial (AudioLabs Erlangen)](https://audiolabs-erlangen.de/resources/MIR/FMP/C8/C8S1_HPS.html) ·
[PCEN + percussive features](https://arxiv.org/pdf/2108.05008).

## Part 4 — Overnight iteration plan (ordered; each measured leakage-safe)

The verifier stays: video-grouped CV, macro-recall ± fold std, plus a
**precision@coverage** curve per subtype (the real target).

| # | Iteration | Lever | Predicted effect | Status |
| --- | --- | --- | --- | --- |
| 1 | TikTok mine → more videos | data | +video count → tighter CIs, more subtypes trainable | 🔄 running |
| 2 | + physics DSP features (envelope/band) | features | rod-vs-lifter ↑ (rate is the physical cue) | 🔄 building |
| 3 | tight-timestamp filter (`tm_range_dur`) | label quality | cleaner labels → higher ceiling | ⏳ |
| 4 | augmentation chain (feature-space → audio-space) | small-data reg. | minority-class recall ↑ | ⏳ |
| 5 | **precision@coverage / abstention** | framing | the actual 99% goal: commit only when confident | ⏳ |
| 6 | hierarchical L1→region→subtype | structure | each level high-precision, errors don't cascade | ⏳ |
| 7 | one-vs-rest specialists for separable subtypes | structure | isolate the *easy* subtypes (belt squeal, wheel-bearing hum) at ~99% precision | ⏳ |
| 8 | HPSS/PCEN re-embed of engine clips | features | transient isolation for knock subtypes | ⏳ |
| 9 | active-learning loop: surface low-confidence + rare clips for human verify | label quality | converts text-labels → gold; raises ceiling | ⏳ |
| 10 | spot-verify text-label precision (sample listen) | measurement | establishes the ~99% *label* number honestly | ⏳ |

## Part 4.5 — Measured results so far (honest, leakage-safe, video-grouped CV)

| Technique | effect | verdict |
| --- | --- | --- |
| more videos (TikTok) | video count 140→416, tighter CIs | ✅ the lever |
| physics DSP features (envelope/band, scale 0.5) | rod-vs-lifter 0.68→0.59 | ❌ hurt (dilutes CLAP) |
| augmentation (mixup+noise, feature space) | region macro 0.748→0.759 | ◐ marginal |
| **tight-label filter** (`tm_range_dur ≤ 45s`) | **region 90%-precision coverage 47%→93%** | ✅ **big** |

**Precision@coverage (the achievable 99% framing):**

| Task | all labels: cov@90% prec | **tight labels: cov@90% prec** |
| --- | --- | --- |
| region (engine vs chassis) | 0.47 | **0.93** |
| part (12-way) | 0.00 | 0.00 (overall 0.26→0.35) |
| knock (rod vs lifter) | ~0 | too few tight clips (32) |

**The decisive lesson (consistent with the headline):** the binding limiter on
the audio model is **label-alignment quality**, not the backbone or the head.
Tight, host-cued labels ("listen to THIS") turn engine-vs-chassis into a usable
90%-precision-at-93%-coverage triage. Loose chapter-span labels poison it.
Per-part one-vs-rest: even wheel bearing (most distinctive) is only 64% precise
when confidently predicted under loose labels — a label problem as much as an
audio one. ⇒ **the path to the goal is maximizing tight, explicit labels (better
mining + more videos), not more model tricks.**

## Part 4.6 — Final full-data results (683 videos, 2,910 high-trust labels)

After mining all of YouTube + TikTok (3,791 sub-labeled clips), with calibrated
video-grouped CV:

| Task | overall acc | cov @ 90% precision | cov @ 95% | precision @ 25% coverage |
| --- | --- | --- | --- | --- |
| **region** (engine vs chassis), tight | 0.85 | **0.78** | 0.23 | **0.94** |
| region, YT tight-timestamp only (n=107) | 0.81 | 0.75 | **0.40** | 0.92 |
| part (12-way) | 0.26 | 0.00 | 0.00 | 0.39 |
| knock (rod vs lifter), all (n=131) | **0.73** | ~0 (noise) | — | 0.66 |

**What more data bought:** knock rod-vs-lifter 0.69→0.73; region coverage held
~0.78 at 90% precision. **What it did not buy:** any high-precision fine-part
classification — 12-way part never reaches 90% precision at any coverage,
robustly across data sizes. The cleanest labels (YT tight-timestamp) give the
best *high-precision tail* (95% precision on 40% of clips) but are scarce (~100).

**Verdict against the 99% goal:**
- ✅ **Label precision** (text side) — the achievable ~99%; verifiable by spot-check.
- ✅ **Region triage** — 90% precision @ ~78% coverage; **94% precision @ 25%
  coverage**. A confident-subset "engine-internal vs chassis" call is real.
- ⚠️ **Fine subtype from audio** — *not* reachable at meaningful coverage with
  current data. Bounded by (a) label-alignment noise [tightening helped most],
  (b) per-class video count [knock = 35 videos], (c) genuine acoustic + non-
  acoustic ambiguity. Honest path: more *tight-timestamp* YT labels (not more
  coarse TT labels) + human verification + possibly non-audio context (RPM).

## Part 5 — Definition of success (what we'll report)

- **Label precision**: a verified estimate (spot-check N clips) that the
  high-trust text labels are ≥ ~95–99% correct. *(This is the achievable 99%.)*
- **Audio model**: per-subtype **precision ≥ 0.9 at the highest coverage we can
  reach**, with honest abstention on the rest — *not* a single accuracy number.
- **A hierarchical triage model + app**: "fault? → region? → if confident,
  specific part + confidence + nearest reference clip; else abstain."
- Every number with a leakage-safe, video-grouped CI. If a subtype can't clear
  the bar, we say so and route it to "needs a mechanic / more data."

The honest bottom line: **99% precision is reachable on the label side and on a
*curated, confident subset* of the audio side; 99% accuracy on every clip from
sound alone is not.** The plan above maximizes the confident subset.
