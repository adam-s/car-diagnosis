# Evaluation methodology — how we know if a change is real

This project makes a falsifiable claim ("the model does X with accuracy Y"), so the
evaluation has to be trustworthy enough that a skeptic can't dismiss it. The
harness that implements all of this is committed at
[`src/cardiag/training/eval/scorecard.py`](../../src/cardiag/training/eval/scorecard.py)
— run `python -m cardiag.training.eval.scorecard` to regenerate
[`docs/SCORECARD.md`](../SCORECARD.md).

## The constraints that drive every choice

- **Small** (~711 clips / 120 videos): single point estimates are dominated by
  sampling noise → everything is repeated cross-validation with a spread, never a
  naked number.
- **Grouped by video**: clips from one recording share mic, vehicle, and ambience.
  If they straddle train/test you measure memorization, not generalization → the
  **video is the atomic split unit**, always (asserted in code).
- **Imbalanced** (74% fault): "always predict fault" scores 0.74 accuracy →
  accuracy-vs-majority is misleading; we use balanced accuracy / MCC / AUPRC and
  always print the majority baseline beside the number.
- **Probabilities feed thresholds and confidence bands** → calibration is a
  first-class metric, not an afterthought.

## The protocol

| Concern | What we do | Reference |
|---|---|---|
| Leakage | `StratifiedGroupKFold` grouped by video; an assertion fails the run if any video appears in both train and test | Kaufman et al., *Leakage in Data Mining*, TKDD 2012; Roberts et al., *Nature Mach. Intell.* 2021 |
| Variance | Repeated (5×5) CV, report mean ± std | Nadeau & Bengio, *Mach. Learn.* 2003 |
| Metrics | Balanced accuracy, macro-F1, **MCC**, AUROC, **AUPRC** — not raw accuracy | Chicco & Jurman, *BMC Genomics* 2020; Saito & Rehmsmeier, *PLOS ONE* 2015 |
| "Above chance?" | By-video **label-permutation null** (shuffle whole-video labels, refit, 200×), report p | Ojala & Garriga, *JMLR* 2010 |
| "Better than last version?" | **Nadeau–Bengio corrected resampled t-test** on per-fold deltas (corrects for fold train-overlap, which the naive t-test ignores → far fewer false "wins") | Nadeau & Bengio 2003; Dietterich, *Neural Comput.* 1998 |
| Calibration | **ECE** (equal-frequency bins) + Brier; **temperature scaling** to fix it | Guo et al., *ICML* 2017; Naeini et al., *AAAI* 2015 |
| Ranked-shortlist heads | **top-k accuracy** (is the true label in the top-k we show?) micro + macro vs random | standard retrieval metric |

## Why each guardrail earned its place here (with the number)

- **Permutation null**: `kind|YouTube` scores balAcc 0.625 against a shuffled-label
  null of 0.50 ± 0.07, **p = 0.035** → the fault/normal signal is real, not an
  artifact of the 74% prior.
- **Corrected-t gate**: the appealing head swaps (PCA-whiten +0.047 on kind,
  prototypical kNN +0.056 on triage) **fail** this test (p > 0.10) — on a corpus
  this small they're within fold-noise. Without the gate we'd have shipped noise as
  progress. This is the single most important guardrail.
- **top-k**: the `cause` head's top-1 balanced accuracy (0.26) reads as "broken,"
  but the product is a ranked top-3 — and the true cause is in that shortlist
  **0.69** of the time (vs 0.25 random). The metric has to match the use.
- **Confound check**: predicting the recording *source* from the embedding scores
  balAcc ≈ 0.65, so a fault/normal head trained on the all-sources split (where
  every normal is YouTube) is partly a source detector → we report the
  **YouTube-only** fault/normal number as the honest one.

## The literature ceiling ("is this good enough to be real?")

DCASE Task 2 (Unsupervised Detection of Anomalous Sounds for Machine Condition
Monitoring) and the in-the-wild automotive analog (Fedorishin et al., *AMPNet*, KDD
2022) put the realistic ceiling for messy, weakly-labeled machine-sound
classification at **mid-70s–low-80s AUROC**, and treat **~90%+ on a confounded
split as a red flag, not a triumph**. Against that ruler:

- **knock** AUROC 0.99 — genuinely strong (a periodic metallic knock is acoustically
  distinctive); above the general ceiling because the task is unusually separable.
- **kind / triage** AUROC 0.70 / 0.72 — right in the credible band; real but modest.
- **cause** top-3 0.69 — a useful ranked shortlist, exactly the documented regime
  for fine-grained fault ID from audio alone.

## The standing conclusion

Across CL2N, PCA-whitening, prototypical/kNN heads, and PANNs fusion, **no
head-architecture change produced a statistically significant accuracy gain** on
the current corpus. The levers that *did* move held-out accuracy are **data
quality and quantity** — label denoising (confident learning), breaking the
source confound (normal clips from TikTok/Reddit), and more videos — consistent
with the learning curve still rising at 100% of the current data. We track every
future change through this scorecard so "it feels better" is never confused with
"it is better."
