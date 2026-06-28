# Converging tiktok/ and youtube/ toward one API

Goal: make the two pipelines call-compatible — same function signatures, same
record schema, same file contracts — so extracting a shared package later is a
file move, not a rewrite. This doc is the target spec plus the lowest-risk
order to get there. It deliberately changes **nothing while batch runs are in
flight**; every step is additive or a rename-at-write-time.

Current divergence at a glance (per stage):

| stage | youtube/ | tiktok/ | divergence |
| --- | --- | --- | --- |
| config | `config.py` | `config.py` | **identical** (byte-for-byte) |
| audio lib | `audio.py` | `audio.py` | TT = YT + thread-safety locks (MPS/Silero). TT is a strict superset |
| discovery | `discover.py` (yt-dlp search) | `discover.py` (stealth browser) + `expand.py` (catalogs) | totally different mechanics — fine; only the **output contract** must converge |
| worklist | `data/worklist.json` (array) | `data/worklist.jsonl` | format + fields differ |
| acquire | `acquire(vid)` → wav (audio-only dl) | `download(url, vid)` → mp4 + `extract_wav(mp4)` | TT needs video for OCR frames |
| per-video | `process_video(vid, title, clap, verbose)` | `process(url, vid, desc, query, clap)` | signatures + responsibilities differ |
| trust gates | `gate()` → status auto/review/reject, tool gate, normal-margin | **none** (raw CLAP top-1 only) | TT must adopt YT's gates |
| cause labels | `l2_from_text(title)` keywords | OCR overlay + (pending) Haiku normalize | complementary, not conflicting |
| batch | `batch.py` (prefetch pool) | `batch.py` (window prefetch) + `batch_shard.py` (sharded) | sharded runner is the keeper |
| enrichment/QA | `vehicle.py`, `validate.py` | `normalize.py` | all three are platform-agnostic once the schema converges |
| LLM calls | `claude -p` JSON-array protocol, copy-pasted | same, copy-pasted | extract once |

---

## 1. Unified record schema (the contract that matters most)

Everything else converges *because* the ledger line converges. Target — one
schema, platform-only fields optional but identically named:

```jsonc
{
  // identity (required, both)
  "clip_id": "7603474706955521310_00",     // f"{video}_{i:02d}" — STABLE, enrichments key on it
  "video": "7603474706955521310",
  "platform": "tiktok",                    // "youtube" | "tiktok"  (YT must start writing this)
  "schema_version": 3,                     // YT writes version:2 today; TT writes none

  // clip bounds + audio file (required; file absent iff status == "reject")
  "start": 2.31, "end": 3.9,
  "file": "data/clips/<video>/clip_00.wav",  // unify on clip_NN.wav for NEW clips; old bNN.wav stays (ledger is authority)

  // sound channel — CLAP L1 (required, both; already identical)
  "l1": "grinding noise", "l1_conf": 0.377, "l1_margin": 0.103,

  // trust gates (required both — TT adopts YT's gate())
  "status": "auto",                        // auto | review | reject  — audio-trust axis
  "mech_confirm": 0.396, "fault_mass": 0.301, "tool_mass": 0.699,
  "tool_evidence": "...",                  // optional, only when tool gate fires
  "video_speech_frac": 0.914,

  // signal features (required, both; already identical)
  "cyclic": {"periodicity": 0.1, "pulse_hz": 10.8, "regularity": 0.36},
  "spectral": {"centroid_hz": 1028.5, "flatness": 0.0012},

  // text channel — cause claims (all optional; either platform may have any)
  "source_text": "If your car is making this noise...",  // YT: title; TT: desc. ONE name.
  "query": "bad wheel bearing sound",      // discovery query (TT has it; YT should carry it through)
  "kind": "fault",                         // fault | normal  (TT: default "fault" — no normal discovery yet)
  "l2_candidates": ["wheel bearing"],      // l2_from_text(source_text) — free for TT, add it
  "ocr_label": "BALL JOINT", "ocr_conf": 0.95,   // TT today; null/absent on YT

  // cause-trust axis (written by normalize step, both platforms)
  "canonical_part": "ball joint",
  "cross_modal_consistent": true,
  "tier": "gold",                          // gold | silver | bronze — ORTHOGONAL to status

  "provenance": {
    "detector": "cascade(v2)+clap-gate",           // YT name; TT adopts
    "models": "silero-vad + CLAP/laion-htsat-unfused",
    "ocr": "applevision",                          // TT only; replaces label_source string
    "banners_stripped": []                         // TT only
  }
}
```

Key decisions encoded above:

- **`status` and `tier` are two orthogonal trust axes**, both present on every
  record. `status` = "do we trust this is a clean mechanical sound" (audio
  gate, from YT). `tier` = "do we trust the cause claim" (text/cross-modal,
  from TT). YT records can earn tiers too (l2_candidates + consistency check);
  TT records need statuses (today 26% of TT clips have l1_margin < 0.1 and
  nothing routes them to review).
- **`source_text` replaces** YT's `provenance.title` and TT's top-level `desc`.
  Every text consumer (l2 keywords, Haiku plausibility, vehicle extraction,
  appliance-domain blocklist) reads one field.
- **`clip_id` format is already identical** (`{video}_{NN}`) — never change it;
  normalize/vehicle/validate enrichments key on it.
- Existing ledgers get a **one-time idempotent backfill** (§5), not in-place
  mutation of running writers.

## 2. Unified function signatures

The platform-specific *implementations* stay different; the *names, arguments,
and return types* converge so the runner and enrichment steps are shared.

```python
# --- worklist item (discovery output contract) -------------------------------
# {"id", "platform", "url", "title", "dur"?, "kind", "query", "author"?}
# "title" is the one name for "the text the platform gives us about the video"
#   (YT title, TT desc); it lands in ledger records as "source_text".
# - YT discover: add platform/url/kind (url is derivable; write it anyway)
# - TT discover/expand: rename desc -> title; kind defaults to "fault"
# - format: JSONL for both (append-friendly, partial-line tolerant). YT's
#   worklist.json array becomes worklist.jsonl.

def discover(...) -> None        # writes data/worklist.jsonl, dedup on id  [per-platform]

# --- acquisition --------------------------------------------------------------
def acquire(item: dict) -> Path  # returns the media file (YT: wav, TT: mp4)  [per-platform]
def get_wav(media: Path) -> Path # ffmpeg 48k mono extraction; identity for YT  [shared]

# --- per-video processing ------------------------------------------------------
def process_video(item: dict, clap: Clap, *, verbose=False) -> list[dict]
# One signature, both platforms. Internally:
#   shared:   get_wav -> candidate_regions(+speech_frac) -> CLAP scores -> gate()
#             -> cyclic/spectral -> l2_from_text(item["title"]) -> records
#   platform hook (optional): frame_labeler(media, regions) -> [(ocr_label, ocr_conf, banners)]
#             TT supplies the Apple-Vision implementation; YT passes None.
# gate() moves from youtube/pipeline.py into the shared lib UNCHANGED — it is
# the most validated code in the repo (every threshold is a measured lesson).

# --- batch running --------------------------------------------------------------
def run_shard(k: int, n: int, *, process_video, acquire) -> None   # [shared]
def merge() -> None                                                # [shared]
# batch_shard.py is the template: stable raw-worklist striding, per-shard
# ledgers, done-set = union(corpus.jsonl, corpus.shard*.jsonl), workers clean
# only their own tmp files, single-thread downloads per worker (platform
# rate-limit budget is a config constant, not an accident of worker count).

# --- LLM helper (extract the 3x copy-paste) -------------------------------------
def llm_json(items: list[dict], instructions: str, *, model=config.HAIKU_MODEL,
             batch=25, timeout=150) -> dict[int|str, dict]
# the `claude -p` + "Reply ONLY a JSON array" + index-keyed merge protocol used
# identically by normalize.py, vehicle.py, validate.py. One place, with a
# dropped-batch counter (today a failed Haiku chunk silently demotes records).

# --- enrichment / QA (platform-agnostic once schema converges) -------------------
def normalize(records) -> records    # cause tiering; reads ocr_label/l2_candidates + source_text
def enrich_vehicles(videos) -> dict  # reads source_text; per-platform meta fetcher injected
def validate(records) -> report      # FFT physics + cluster coherence + Haiku plausibility
```

Correction to the worklist comment above (kept inline to show the reasoning):
the unified text field is **`title`** in worklist items and **`source_text`**
in ledger records — worklists describe the video, records describe the claim
context. TT's `desc` maps to both.

## 3. Shared package shape (the eventual extraction)

```text
corpuslib/                      # name TBD
  config.py                     # today's shared config + per-platform query sets
                                #   (move TT's PROBLEM_QUERIES out of discover.py)
  audio.py                      # tiktok's version verbatim (thread-safe superset)
  gates.py                      # gate(), l2_from_text()  (from youtube/pipeline.py)
  ocr.py                        # ocr_raw/ocr_consensus/frame_at  (mac-only optional extra)
  llm.py                        # llm_json()
  ledger.py                     # worklist/ledger IO, done-set, schema_version, backfill
  runner.py                     # sharded batch runner + merge
  enrich/
    normalize.py  vehicle.py  validate.py
youtube/   -> thin entrypoints: discover.py + platform acquire + a 10-line main
tiktok/    -> thin entrypoints: discover.py, expand.py + platform acquire/frame_labeler
```

Packaging notes:

- Scripts are PEP-723 single-file today. Keep that for entrypoints; the shared
  lib becomes a real package (uv workspace member or plain `pyproject.toml`)
  the entrypoints depend on via `tool.uv.sources` path dependency.
- `ocrmac` (Apple Vision) and `patchright` are heavy, platform-bound deps —
  optional extras (`corpuslib[ocr]`, `corpuslib[stealth]`), never imports of
  the core.
- macOS-only OCR is acceptable: it's a labeling-time tool, not a training-time
  dependency.

## 4. Per-platform gaps the convergence exposes (do these, they're cheap)

1. **TT adopts `gate()`** — biggest data-quality win; brings status
   auto/review/reject, the tool gate, and the normal-margin rule to TikTok.
   Can run retroactively: all gate inputs except `mech_confirm`/`fault_mass`/
   `tool_mass`/`speech_frac` are already in TT records; a backfill pass
   re-scores existing clip wavs (they're all on disk) with the two extra CLAP
   prompt sets — no re-download needed.
2. **TT gets `l2_candidates` for free** — run `l2_from_text` over `desc`/query.
3. **YT gets `tier` for free** — `normalize()` over `l2_candidates` +
   `source_text` (no OCR needed; keyword claims are weaker than overlay OCR,
   so cap YT at silver unless OCR is added later — record the cap in config).
4. **TT has no `kind:"normal"` discovery** — the unified worklist makes this
   visible; add normal-baseline queries to TT discovery or accept YT as the
   sole normals source (fine — domain-matching argues for it anyway; decide
   and write it in config).
5. **normalize must read `source_text`** — closes the dryer/appliance gold
   contamination hole found in the corpus review (Haiku currently judges OCR
   text with no context); add the domain blocklist at the same time.

## 5. Migration order (each step safe alone; nothing touches a live run)

| step | what | risk |
| --- | --- | --- |
| 0 | Freeze this spec; **wait for in-flight TT batch to finish** | none |
| 1 | `cp tiktok/audio.py youtube/audio.py` (thread-safe superset; YT API unchanged) | none |
| 2 | Extract `llm.py` helper; point normalize/vehicle/validate at it | tiny |
| 3 | Move TT `PROBLEM_QUERIES` into the shared `config.py`; re-sync the two copies | none |
| 4 | **Backfill script** (`temp/` or `tools/`): rewrite both ledgers to the §1 schema — add `platform`/`schema_version`/`kind`/`source_text`, fold `provenance.title`→`source_text`, `label_source`→`provenance.ocr`, TT `desc`→`source_text`. Idempotent (skips records already at schema_version 3); writes `corpus.v3.jsonl` then atomically replaces; original kept as `.bak` | low — pure JSONL transform, clip files untouched |
| 5 | TT gate backfill (re-score existing clip wavs with CONFIRM/FAULT/TOOL prompts → `status` + masses) | low — additive fields |
| 6 | Update both `batch`/`pipeline` writers to emit the v3 schema natively; unify new-clip filenames on `clip_NN.wav` | low |
| 7 | Generalize `batch_shard.py` into the shared runner; YT batch becomes a caller of it | medium — test on a 20-video slice |
| 8 | Extract `corpuslib/`, thin out the platform dirs | mechanical by now |

Steps 1–5 require no coordination between the pipelines and can happen the
moment the current TikTok run completes. Step 4's backfill is the gate for
sharing enrichment code; everything after is convenience.

## 6. Invariants the framework must never break

- **`clip_id` stability** — enrichments and dedup key on it.
- **Restart-safety contract**: a video is done iff its id appears in the
  ledger (or a shard); writers append one JSON line per clip and flush per
  video; readers tolerate a torn final line.
- **The ledger `file` field is the authority on clip paths** — never derive
  paths from naming conventions (two conventions already exist on disk).
- **Rejects carry no wav** (metadata-only records are expected; 25% of YT).
- **Expensive models never in-loop**: Haiku batched post-hoc, Fable only as
  the one-time gold auditor (config.HAIKU_MODEL is the only LLM constant).
- **Download gentleness is a global budget per platform** (YT: 6 concurrent;
  TT: ~4), owned by config, independent of worker count.
