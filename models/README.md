# Pre-trained model (optional, opt-in)

These two small artifacts are a **pre-trained cardiag model** you can use instead
of training your own:

- `best_model_clap.joblib` — the kind / knock / cause heads (temperature-calibrated)
- `triage_model.joblib` — the engine-vs-running-gear triage head

Use it by pointing the server at this directory:

```
cardiag serve --model models
```

Without `--model`, the app uses whatever you trained yourself with `cardiag train`
(or none) — the default clone-and-train flow is unchanged. This model is an
**explicit opt-in**, which is why it lives here and not in `data/training/`.

## Provenance & honesty

This model was **not** trained on the small corpus a fresh clone scrapes. It was
built by `scripts/build_shipped_model.py` from a larger labeled corpus the author
already had (~8.8k clips, ~16k CLAP embeddings) — **not** bundled here. Those weak
labels are noisy, so the build **cleans the noisiest ~10% per source** (confident
learning) before fitting; cleaning measurably improves accuracy on a held-out
**human-verified** set, which is how the corpus's quality is judged rather than
trusted. So:

- It is **stronger** than the from-scratch model — fault-vs-normal balanced
  accuracy **0.73 ± 0.02** (vs ~0.63 from the 711-clip self-scrape) and, on the
  **1,355-clip human-verified eval set, 0.69 balanced / 0.77 AUROC** — right at the
  literature ceiling for in-the-wild machine sound.
- The fine **cause** head stays weak (it's a 12-way label that's hard from audio
  alone); trust it only as the ranked top-3 shortlist, and lean on the fault /
  knock / triage heads.
- It is **not reproducible from a clone** (the corpus isn't included). The numbers
  in [`docs/SCORECARD.md`](../docs/SCORECARD.md) and [`docs/RESULTS.md`](../docs/RESULTS.md)
  remain the honest, clone-reproducible ones from the self-scraped corpus.

Only the ~100 KB linear-head artifacts are committed — no audio, no embeddings.

## Security note

joblib executes code on load. Only load model files you trust (these were produced
by the build script in this repo).
