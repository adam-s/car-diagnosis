# car-diagnosis — documentation

`car-diagnosis` identifies a car's mechanical fault from the **sound it makes**.
The whole loop is self-contained and reproducible from a fresh clone — **scrape →
clean → segment → label → train → serve** — needing only `yt-dlp` + CLAP (no LLM,
no external datasets, no bundled model).

The honest one-paragraph version: sound *type* (grind / squeal / knock / whine /
tick) is decidable from audio; the exact *cause/part* mostly is not. So the system
trains a CLAP-embedding + linear-head classifier whose real, validated outputs are
strong **knock** detection, a calibrated **fault-vs-normal** and **engine-vs-
running-gear** triage, and a ranked **top-3 cause** shortlist — never an
overconfident single guess. That honesty, and the rigorous evaluation that backs
it, is the point.

## Start here

- **[../README.md](../README.md)** — install and run it (`cardiag start`).
- **[WALKTHROUGH.md](WALKTHROUGH.md)** — a guided first run, end to end.
- **[scraping-guide.md](scraping-guide.md)** — start-to-finish scraping of YouTube,
  TikTok, and Reddit with the Camoufox stealth browser.
- **[architecture.md](architecture.md)** — Mermaid diagrams of the pipeline,
  scrapers, cleaning cascade, training, and inference.

## Results and evaluation

- **[RESULTS.md](RESULTS.md)** — measured, by-video cross-validated numbers from a
  real run, and which improvements actually helped.
- **[SCORECARD.md](SCORECARD.md)** — the auto-generated scorecard
  (`python -m cardiag.training.eval.scorecard`).
- **[MODEL_CARD.md](MODEL_CARD.md)** — per-head metrics, intended use, limitations.
- **[research/eval-methodology.md](research/eval-methodology.md)** — how we tell a
  real improvement from noise (grouped CV, permutation null, calibration), with
  literature citations.
- **[PROOFS.md](PROOFS.md)** — the verification log: what is and isn't provable.
- **[research/iteration-research.md](research/iteration-research.md)** — research
  notes behind the design choices.
