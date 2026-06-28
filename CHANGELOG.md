# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [Unreleased]

### Added
- `cardiag doctor` (env preflight), `cardiag inspect` (visual HTML report),
  `cardiag gallery` (audio corpus grid), `cardiag train --fixtures` (offline,
  no-network training in ~2s), `cardiag start` onboarding.
- Camoufox stealth-Firefox transport for TikTok + Reddit scraping; YouTube via
  yt-dlp. One unified `_label_audio` path feeds all three sources.
- Bundled fixture embeddings for offline training/tests; snapshot regression
  harness; robustness + honesty test suites. ruff + mypy + pre-commit + CI.

### Changed
- **Train/serve embedding skew removed.** Training and inference now share one
  embedding contract (`cardiag.audio.embed`): every vector a head sees — a corpus
  clip at train time, an isolated span at serve time — is the *same* single-span
  `embed_clip()` output. Multi-span recordings are pooled in **probability space**
  (mean of each head's `predict_proba` over spans) instead of by averaging
  embeddings, which produced a renormalized vector the scaler/LogReg never saw at
  fit time. `triage` now cleans + pools like `diagnose` (previously embedded the
  whole raw file). New `tests/test_embedding_contract.py` locks the invariant.

### Fixed (hardening pass)
- **Data loss:** a flaky/zero-yield re-scrape no longer truncates the corpus —
  `corpus.jsonl` is now written atomically and merged (deduped by clip id).
- **Honesty:** a degenerate (single-class) model no longer reports a confident
  verdict; `diagnose` returns UNCERTAIN with a note, `triage` abstains, and
  training refuses to write an all-constant model.
- **Crashes:** bad/missing/non-audio inputs and invalid model files produce clear
  one-line messages (CLI) or HTTP 400/413 (web), never raw tracebacks or 500s.
- **Web:** upload size cap + chunked read, suffix allowlist, serialized inference,
  guaranteed temp-file cleanup, non-loopback `--host` warning.
- **Security:** `--` end-of-options before scraped URLs in yt-dlp calls;
  configurable cookie browser (`CARDIAG_COOKIES_BROWSER`); model files documented
  as trusted-input-only (joblib executes code on load); fixtures load without
  pickle.

## [0.1.0]
- Initial extraction of the sound pipeline (scrape → clean → label → train →
  classify) into the installable `cardiag` package.
