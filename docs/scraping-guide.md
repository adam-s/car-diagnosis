# Scraping, Start to Finish: Camoufox as the Stealth Foundation

How to go from an empty `data/` directory to a labeled corpus, using
[Camoufox](https://camoufox.com/) wherever a real, fingerprint-clean browser is
needed. Camoufox is a hardened Firefox whose anti-fingerprinting lives at the
**C++ engine level** (not patched-in JavaScript), so it passes WAFs that detect
Chromium automation and never leaks the SwiftShader/`HeadlessChrome` tells.

A Camoufox wrapper already exists, so we are not writing browser automation from
scratch. We lift `scraper/camoufox_scraper/browser.py` (plus its `antibot/`
solvers) and point it at the media platforms.

> **Where Camoufox fits.** YouTube and Reddit don't need a stealth browser:
> `yt-dlp` and plain HTML are enough. TikTok does. Its search feed is gated and
> Chromium automation gets flagged. The TikTok discovery step in the source
> projects currently uses `patchright` (stealth Chromium). The standardization here
> runs that same network-interception pattern through Camoufox, because we already
> maintain the wrapper and Firefox passes these gates more reliably. One browser
> layer, one set of antibot code, reused everywhere.

---

## 0. The Camoufox wrapper we already have

Lift from `*/scraper/camoufox_scraper/`. The pieces:

- **`browser.py`** — a context-manager `Browser` around `camoufox.sync_api.Camoufox`.
  Gives you a Playwright-compatible `page` with all the stealth defaults applied.
- **`transport.py`**: `http_json()` for open APIs plus a `Ja4GuardError` that
  refuses to send a protected request over plain HTTP (it forces the request
  through the browser so it carries real Firefox TLS).
- **`antibot/datadome.py`**, **`antibot/turnstile.py`**: challenge solvers. These
  are only needed for the parts scraper; the media platforms here don't hit them,
  but the code is there if a target adds one.

Key knobs on `Browser`, with the lessons baked in:

```python
Browser(
    headless="auto",     # → "virtual" (Xvfb) on Linux, headed on macOS. NEVER True.
    warm=False,          # visit neutral pages first so a fresh persona has history
    profile=None,        # set a name to persist cookies/cf_clearance across runs
    persona_os="macos",  # pin OS so the UA stays stable for a persistent profile
    proxy=None,          # only needed at high concurrency
)
```

Rules encoded in this wrapper, learned the hard way:

1. **Never run true `headless=True`.** WAFs block it regardless of browser. On a
   Linux server use `headless="virtual"` (Camoufox's built-in Xvfb); on macOS dev
   run headed. `"auto"` picks correctly.
2. Use a fresh persona per session by default. Each launch randomizes the
   fingerprint at the engine level, undetectable by JS. If a session gets flagged,
   the recovery is to start a new one.
3. Warm history makes a fresh persona look real. `warm=True` browses a few neutral
   pages (Wikipedia, a subreddit, a search) before the target.
4. Use persistent profiles only when a cookie needs to survive (e.g. a solved
   challenge). Then you must pin `persona_os`, or the UA drifts and the cookie is
   invalidated.
5. JA4 tiering. Open APIs use plain HTTP (`http_json`). Gated endpoints use in-page
   `fetch()` inside Firefox so the request carries real Firefox's TLS/HTTP2
   fingerprint. The `Ja4GuardError` enforces this so you can't get it wrong.

Resource budget (from production runs): ~1.4 GB RAM per browser instance, ~1 GB
disk for the Camoufox Firefox + venv. A 2–4 vCPU / 4–8 GB Linux box handles normal
volume with 1–2 instances. No proxy needed until you scale concurrency.

---

## 1. YouTube — `yt-dlp`, no browser needed

YouTube search and download go through `yt-dlp` directly.

```
ingest/youtube/discover.py   # yt-dlp ytsearch over fault + matched-normal queries
                             #   -> data/youtube/worklist.json
ingest/youtube/batch.py      # per video: download audio, run the clean+segment
                             #   cascade, CLAP-confirm, write clips + corpus.jsonl
ingest/youtube/capture.py    # yt-dlp: chapters, --write-auto-subs transcript, comments
ingest/youtube/enrich.py     # join chapter/transcript text down to each clip by time
```

Flow:

1. `discover.py` runs curated **fault** queries and matched normal queries (normal
   is earned, not assumed) into a worklist of video ids.
2. `batch.py` is a restart-safe controller: network (download), CPU (cascade), and
   GPU (CLAP) pipelined, skipping anything already in the ledger.
3. `capture.py` pulls the free native metadata (chapters, auto-captions, top
   comments). `enrich.py` joins it to clips by timestamp.

This stage alone produced ~9,000 clips in the source project. It needs only
`yt-dlp` + `ffmpeg`.

## 2. TikTok — Camoufox stealth interception, then `yt-dlp`

TikTok is where Camoufox pays off. Don't parse the page. Let the page make its own
authenticated API call and **intercept the JSON response**.

```
ingest/tiktok/discover.py  # Camoufox: open search, intercept /api/search/item/full/
                           #   -> data/tiktok/worklist.jsonl
ingest/tiktok/expand.py    # yt-dlp: expand on-topic accounts' catalogs (374 -> 1740)
ingest/tiktok/batch.py     # per video: yt-dlp mp4 -> ffmpeg frames -> OCR consensus
                           #   -> ffmpeg wav -> clean cascade -> CLAP L1 -> corpus.jsonl
```

Discovery pattern (port the existing patchright interception onto the Camoufox
`page`):

```python
from camoufox_scraper import Browser

hits = []
with Browser(warm=True) as b:            # fresh persona, warmed history, virtual headless on Linux
    b.page.on("response", lambda r:                       # intercept, don't scrape
        hits.append(r.json()) if "/api/search/item/full/" in r.url else None)
    b.page.goto("https://www.tiktok.com/search?q=car+grinding+noise",
                wait_until="domcontentloaded")
    b.page.mouse.wheel(0, 4000)          # scroll to trigger more feed pages
    # ... collect video ids/urls from the intercepted feed JSON
```

Why this works and is stable:

- Camoufox presents a real Firefox fingerprint, so the search endpoint serves the
  same JSON it serves a human. No HTML parsing, no brittle selectors.
- `warm=True` plus a fresh persona keeps it unsupervised and scalable.
- The expensive part (download + OCR) is deferred. `expand.py` first multiplies
  reach for free by pulling whole catalogs of on-topic accounts with `yt-dlp`.

Then `batch.py` does the per-video work. This OCR-label step is what makes TikTok
uniquely valuable:

1. `yt-dlp` downloads the mp4.
2. `ffmpeg` extracts ~3 frames spread across the clip.
3. `ocrmac` (Apple Vision) OCRs each frame. A **temporal consensus** keeps text
   that recurs across frames (the burned-in fault label) and drops one-off
   narration. Banners and channel names are stripped.
4. `ffmpeg` extracts the audio to WAV, the clean+segment cascade runs, and CLAP
   assigns the L1 sound type.
5. A clip is **gold** only when the OCR'd part and the CLAP sound type agree: two
   independent modalities corroborating each other.

## 3. Reddit — DEPRECATED as a training source

> **Deprecated.** Reddit posts are long, uncurated, and off-target, which adds
> label noise without improving accuracy. Dropping Reddit slightly raised the
> headline number. Train on YouTube+TikTok instead; the cleaning cascade handles
> messy input at inference time. The code below still works (`cardiag scrape reddit`
> warns and proceeds) but is not part of the recommended loop.

Reddit's `.json` API 403s under load, but the old HTML site stays open and
auth-light. Extraction is deterministic regex. The only authenticated step is
`yt-dlp` reusing your Firefox cookies to pull `v.redd.it` audio.

```
ingest/reddit/scrape.py  # paginate old.reddit /{new,top,hot}/, regex the post attrs,
                         #   yt-dlp --cookies-from-browser firefox for audio
                         #   -> data/reddit/posts.jsonl  (title, selftext, top comments)
ingest/reddit/pipeline.py
```

What it does (already implemented, see `ingest/reddit/scrape.py`):

- Walks diagnosis-focused subs (`MechanicAdvice`, `AskMechanics`, `carproblems`,
  `autorepair`, `Justrolledintotheshop`) across `new`/`top`/`hot` feeds, following
  the "next" button.
- Parses each post tile via `data-fullname` / `data-url` / `data-permalink` /
  `data-domain`, keeping only video domains (`v.redd.it`, YouTube, Streamable).
- Dedups reposts across subs by a stable media id.
- `yt-dlp --cookies-from-browser firefox -x --audio-format wav` pulls the audio at
  48 kHz mono; only clips 2–180 s with a real audio codec are kept.
- Captures the post title, self-text, and top-8 comments (score-sorted, automod
  filtered) as the text signal.
- Polite 3.5 s throttle, restart-safe via the JSONL ledger.

No browser is needed here. If Reddit ever tightens HTML access, the same Camoufox
`Browser` drops in as the transport with zero other changes.

---

## 4. After scraping: one shared pipeline

All three platforms write `corpus.jsonl` rows with the same shape (clip path,
candidate L1 sound type, raw text signals). From there everything converges on the
shared, platform-agnostic stages:

```
pipeline/fusion.py      # LLM fuses audio + text signals -> {cause, kind, confidence}
pipeline/gate_tier.py   # objective trust tiering: gold / silver / bronze
pipeline/music_gate.py  # CLAP music filter -> exclude contaminated clips
```

…and then into training (see [`architecture.md`](architecture.md) for the full
pipeline diagram).

---

## Quick-start command sequence

```bash
# one-time: install the Camoufox Firefox (matches the pinned playwright==1.51.0)
uv run python -m camoufox fetch

# YouTube  (yt-dlp; biggest volume)
uv run ingest/youtube/discover.py
uv run ingest/youtube/batch.py
uv run ingest/youtube/capture.py && uv run ingest/youtube/enrich.py

# TikTok   (Camoufox discovery -> yt-dlp + OCR)
uv run ingest/tiktok/discover.py
uv run ingest/tiktok/expand.py
uv run ingest/tiktok/batch.py

# Reddit   (old.reddit HTML -> yt-dlp)
uv run ingest/reddit/scrape.py 12

# fuse + tier the combined corpus
uv run pipeline/fusion.py
uv run pipeline/gate_tier.py
uv run pipeline/music_gate.py
```

Every step is restart-safe (JSONL ledgers, skip-if-seen), so you can stop and
resume any stage without re-doing work.

---

## Legal / ethical note for the open-source README

This pulls publicly visible content for **research and model training**. The README
should: respect each platform's terms and robots, keep the polite rate limits the
code already enforces, store only what's needed (audio + text labels, not personal
data), and ship the corpus as fetch scripts rather than redistributing copyrighted
media, exactly how `external-data/fetch.sh` already handles the public reference
datasets.
