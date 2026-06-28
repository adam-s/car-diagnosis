# car-diagnosis — Documentation

`car-diagnosis` identifies a car's mechanical fault from the **sound it makes**. It
is a focused, open-source extraction of the sound pipeline from two larger projects:
scrape public car-fault audio, clean and segment it, label it with OCR/text/
transcription metadata, train a calibrated classifier, and serve inference through a
shared core that powers both a CLI and a local web app.

Scope is deliberately narrow — **scrape → clean → segment → label → train →
classify**. No parts pricing, no browser extension, no chat agent.

## Read in order

1. **[00-source-projects.md](00-source-projects.md)** — what the two parent projects
   (`detect-mech-issues`, `detect-study`) are, and what this project takes from each.
2. **[01-what-we-have.md](01-what-we-have.md)** — the parts list: every stage that
   already exists, the exact library/model it uses, and where to lift it from.
3. **[02-scraping-with-camoufox.md](02-scraping-with-camoufox.md)** — start-to-finish
   scraping of YouTube, TikTok, and Reddit, with Camoufox as the stealth-browser
   foundation.
4. **[03-architecture-diagrams.md](03-architecture-diagrams.md)** — Mermaid diagrams
   of the full pipeline, scrapers, cleaning cascade, training, and inference.

## The honest one-paragraph summary

Sound *type* (grind / squeal / knock / whine / tick) is decidable from audio; the
*cause/part* mostly is not — it comes from text. So the system fuses audio with
scraped text labels to build a trustworthy corpus, then trains a CLAP-embedding +
linear-head classifier whose real, validated output is a coarse fault verdict
(fault-vs-normal, engine-internal-vs-running-gear) with a **calibrated** confidence
band — not an overconfident single-cause guess. That honesty is the point.
