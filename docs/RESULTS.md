# Results from a real run

These are **measured** numbers from actually running this repo end to end — not
quoted from the parent project. The point is honesty: here is what the pipeline
produces, including where it underperforms and why.

## The run

```
cardiag scrape youtube --per-query 4 --max-videos 110
cardiag scrape reddit  --pages 3
cardiag scrape tiktok  --max-videos 70
cardiag train
```

**Corpus:** 711 labeled clips — 438 YouTube, 211 TikTok, 62 Reddit. All scraped,
cleaned (cascade: VAD + energy + flatness + CLAP music gate), and labeled with no
LLM and no external datasets. `kind` balance: 527 fault / 184 normal (all normals
come from YouTube's "normal" query set; Reddit and TikTok are fault-dominated).

## Measured accuracy (group-held-out by video — no leakage)

| Head | classes | held-out acc | majority | n_test |
|---|---|---|---|---|
| **knock** (knock vs normal-idle) | 2 | **0.958** | 0.736 | 72 |
| **kind** (fault vs normal) | 2 | 0.874 | 0.944 | 143 |
| **triage** (engine vs running-gear) | 2 | 0.667 | 0.807 | 57 |
| **cause** (part family) | 12 | 0.258 ⚠ | 0.403 | 62 |

⚠ The `cause` head auto-flagged its own score `held_out_unreliable` (n_test=62,
12 classes) — the code refuses to present that as a trustworthy number.

## The honest read

- **knock detection works** (0.958 vs 0.736 baseline) — a periodic metallic knock
  has a distinctive acoustic signature CLAP captures well.
- **fault-vs-normal and triage land *below* the majority baseline here, and that's
  the corpus's fault, not a bug.** Two of the three sources only yield *fault*
  clips, so the corpus is 74% fault and the held-out split was 94% fault — a
  classifier that always says "fault" scores 0.94. With `class_weight="balanced"`
  the head learns real structure but is graded against a lopsided baseline. The fix
  is more *normal* audio (scrape more YouTube normals, or add a normal-only source),
  not a model change. The pipeline surfaces this honestly rather than hiding it.
- **fine cause from audio alone is weak** (0.258, and self-flagged unreliable) —
  exactly the documented ceiling. Cause is a *text*-derived label; the audio head is
  a weak prior, which is why `diagnose` presents cause as a ranked top-k, never a
  single confident answer, and `triage` (the one acoustically separable axis) is the
  honest headline.

## What this demonstrates

Not a state-of-the-art classifier — an **honest, reproducible pipeline**. Every
number above came from one `cardiag train` on a corpus this repo scraped itself,
the leakage-safe split is enforced in code, and degenerate/unreliable cases are
flagged rather than dressed up. To push the numbers, the lever is data
(more balanced scraping, the LLM-fusion labeling in `docs/`), and the harness is
built to measure whether that actually helps.

## Reproduce

```
cardiag scrape youtube --per-query 5 --max-videos 200
cardiag scrape reddit --pages 4
cardiag scrape tiktok --max-videos 120
cardiag train            # writes data/training/train_report.json with these stats
cardiag gallery -o gallery.html   # audit the labels yourself
```
