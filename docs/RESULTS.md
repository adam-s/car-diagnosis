# Results from a real run

Measured numbers from running this repo end to end — not quoted from a parent
project. Everything here is reproduced by the committed scorecard
(`python -m cardiag.training.eval.scorecard` → [SCORECARD.md](SCORECARD.md)); the
methodology and citations are in [research/eval-methodology.md](research/eval-methodology.md),
and the per-head summary is the [MODEL_CARD.md](MODEL_CARD.md).

## The run

```
cardiag scrape youtube --per-query 4 --max-videos 110
cardiag scrape reddit  --pages 3
cardiag scrape tiktok  --max-videos 70
cardiag train
```

**Corpus:** 711 labeled clips, 120 videos — 438 YouTube, 211 TikTok, 62 Reddit;
527 fault / 184 normal. Scraped, cleaned (cascade: VAD + energy + flatness + CLAP
music gate), labeled with no LLM and no external datasets.

## Measured accuracy — by-video StratifiedGroupKFold, 5×5 repeats (no leakage)

Balanced accuracy (chance = 0.50) and AUROC, grouped by video so clips from one
recording never straddle train/test. The realistic ceiling for in-the-wild
machine-sound classification is mid-70s–low-80s AUROC (DCASE Task 2); ~90%+ on a
confounded split is a red flag, not a triumph.

| Head | balAcc | AUROC | how to read it |
|---|---|---|---|
| **knock** (knock vs normal-idle) | **0.94** | **0.99** | strong — distinctive periodic signature; the standout |
| **triage** (engine vs running-gear) | 0.66 | 0.72 | real, modest — the honest *headline product* |
| **kind** (fault vs normal, YouTube-only) | 0.63 | 0.70 | real (permutation **p=0.035**), modest |
| **cause** (part family, **top-3 shortlist**) | — | — | **true cause in top-3 = 0.69** (top-4 0.76) vs 0.25 random |

`cause` returns a ranked top-3 with confidence, never one forced answer — so the
honest metric is whether the true part is *in the shortlist* (0.69), not the harsh
top-1 (0.40). That's a genuinely useful capability the single-answer number hides.

## The honest read

- **knock detection is genuinely strong** (AUROC 0.99) and the one head you can act
  on alone.
- **triage and fault/normal are real but modest** — above chance (kind permutation
  p=0.035) and right in the literature's credible band, not impressive. They are
  **data-limited, not broken**: the learning curve is still rising at 100% of the
  current videos.
- **the corpus has a source confound** — every `normal` clip is YouTube, while
  TikTok/Reddit are 100% fault, so the recording source is partly predictable from
  audio (balAcc ≈ 0.65). We report the **YouTube-only** fault/normal number (source
  held constant) as the honest one; the all-sources AUROC (0.77) is confound-inflated.
- **probabilities are calibrated** (temperature scaling): `kind` ECE 0.317 → 0.044,
  so a weak head reads UNCERTAIN instead of emitting confident wrong verdicts.

## What we tried, and what actually moved the needle

Tracked through the scorecard with a Nadeau–Bengio corrected-t significance gate:

| Change | Result |
|---|---|
| CL2N, PCA-whitening, prototypical/kNN heads | **no significant gain** (within fold-noise, p > 0.10) |
| CLAP + PANNs embedding fusion | **no gain** (all deltas within noise) |
| CLAP zero-shot relabeling of causes | **hurts** (zero-shot only 17.5% agrees with keyword labels) |
| Temperature-scaled calibration | **adopted** — large ECE reduction, decision-preserving |
| Refit shipped model on 100% of data | **adopted** — was discarding 25% |
| Label denoising (confident learning) | **promising** (held-out balAcc +0.04–0.06; see scorecard history) |

**The binding constraint is data, not model architecture.** The levers that move
held-out accuracy are: scraping `normal` clips from TikTok/Reddit (breaks the
confound), denoising weak labels, and more videos. The harness is built to measure
whether any of those actually helps, fold-by-fold.

## What this demonstrates

Not a state-of-the-art classifier — an **honest, reproducible, self-measuring
pipeline**. Every number above comes from one `cardiag train` on a corpus this repo
scraped itself, the leakage-safe split is enforced in code, weak/degenerate cases
are flagged rather than dressed up, and the evaluation is rigorous enough to tell a
real improvement from noise.

## Reproduce

```
cardiag scrape youtube --per-query 5 --max-videos 200
cardiag scrape reddit --pages 4
cardiag scrape tiktok --max-videos 120
cardiag train                                   # writes data/training/train_report.json
python -m cardiag.training.eval.scorecard       # writes docs/SCORECARD.md
cardiag gallery -o gallery.html                 # audit the labels yourself
```
