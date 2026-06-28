# Plan — Full per-video metadata capture

**Goal:** for *every* discovered video (746 YouTube + 1,740 TikTok), capture all text/metadata
once, losslessly, then join it down to the clip level so each `.wav` inherits real, local labels
instead of one truncated title.

**Status quo (why this is needed):** the L2 fault label currently rides on the video *title* alone.
Transcripts, descriptions, tags, chapters, and comments were never fetched. See
`temp/` analysis and `.claude/reference/research.md`.

---

## What the probe proved (video `ZrWfYh_9TZo`)

One `yt-dlp --skip-download` call returns, with **no video download**:

- **info.json** — `title`, `description` (1572 chars), `tags` (40), `channel`, `upload_date`,
  `view/like/comment_count`, and **`chapters`** = `[{start, end, title}]`.
- **`chapters`** → `Wheel Bearing 378–663s`, `Engine Failure 663–932s`. **Timestamped fault labels,
  authored by the uploader.** This is the single highest-value field — near-ground-truth, free.
- **Transcript** — `en.json3` = 1,179 **timestamped** cues (`tStartMs` + text). Native captions when
  present, auto-subs otherwise.
- **Top comments** — embedded in info.json via `comment_sort=top;max_comments=N` (per instruction:
  **top comments only**, not the full thread). Often contain a second diagnosis.

So the capture is cheap (text only) and the payoff (chapters + aligned transcript) directly fixes the
weak-label problem.

---

## Resources

- **8 free CPUs** → run yt-dlp fan-out at `-P 8` (network-bound; 8 concurrent is the cap).
- **GPU available** → `faster-whisper large-v3` on GPU for the caption-less tail and for all TikToks
  (TikTok has no captions). Batchable, fast on GPU.

---

## Storage layout (idempotent, resumable)

```
youtube/data/meta/<video_id>/
    info.json            # full yt-dlp metadata incl. top comments + chapters
    *.json3              # timestamped subtitle tracks (native + auto)
    whisper.json         # ONLY if no usable captions: faster-whisper output
tiktok/data/meta/<video_id>/
    info.json            # full untruncated desc, hashtags, author, music, stats, top comments
    whisper.json         # whisper on the clip wav(s) for spoken TikToks
```

Resumability: skip any `<id>/` that already has a non-empty `info.json` (and, if transcript was
expected, a subtitle/whisper file). Re-runnable any time as new videos get discovered.

---

## Steps

### 1. YouTube metadata + transcript + top comments — `youtube/capture.py`
For each of the 746 worklist IDs, run (parallel `-P 8`):
```
yt-dlp --skip-download --no-warnings --ignore-errors \
  --write-info-json \
  --write-subs --write-auto-subs --sub-langs "en.*" --sub-format "json3/vtt" \
  --write-comments \
  --extractor-args "youtube:comment_sort=top;max_comments=20,20,0,0" \
  -o "youtube/data/meta/%(id)s/%(id)s.%(ext)s" \
  -a youtube/urls.txt
```
- Retry/backoff on throttle; log failures to `meta/_failed.txt`.
- Est: ~746 vids / 8-way / ~3s each ≈ a few minutes.

### 2. Whisper fallback (GPU) — `transcribe.py`
- Videos with **no** usable caption track → download audio-only (`-f bestaudio -x`) to a temp file,
  run `faster-whisper large-v3` (GPU, `word_timestamps=True`), write `whisper.json`, delete audio.
- Keeps everything timestamped so it aligns the same way as captions.

### 3. TikTok metadata — `tiktok/capture.py`
- `yt-dlp --skip-download --write-info-json --write-comments
   --extractor-args "tiktok:..." ` per video → full `desc`, hashtags, `author`, `music`, stats,
   top comments. If yt-dlp gets blocked, fall back to the existing patchright stealth path in
   `tiktok/discover.py` to pull the same JSON.
- No captions on TikTok → Step 2 whisper runs on the clip `.wav`s already on disk.
- Parse `#hashtags` out of `desc` into a `tags` list.

### 4. Consolidate + clip-level join — `enrich.py`  ← the payoff
Produce two artifacts per platform:
- `meta_corpus.jsonl` — one row/video: all text fields normalized
  (`title, description, tags, chapters, channel, stats, top_comments, transcript[]`).
- `corpus.enriched.jsonl` — re-emit every existing clip row, **joined** to its video metadata by
  `[start,end]` overlap:
  - `clip_transcript` = transcript cues overlapping the clip window (± small pad).
  - `chapter_label` = title of the chapter containing the clip midpoint (e.g. `Wheel Bearing`).
  - `video_tags`, `video_description`, `top_comments` attached.
  - **`l2_local`** = re-derive fault category from `chapter_label` + `clip_transcript` (local
    narration) instead of the global title. Keep old `l2_candidates` for comparison.

### 5. Measure the lift
- Report: how many clips now get a `chapter_label`; how many `l2_local` disagree with the old
  title-derived `l2`; per-class counts before/after; coverage of transcript vs whisper.
- This number tells us how much the labels actually improved.

---

## Field inventory captured (vs. before)

| Field | Before | After |
|---|---|---|
| title | ✅ (truncated) | ✅ full |
| description | ❌ | ✅ |
| tags / hashtags | ❌ | ✅ |
| **chapters (timestamped)** | ❌ | ✅ **fault labels** |
| **transcript (timestamped)** | ❌ | ✅ captions + whisper |
| top comments | ❌ | ✅ (top N only) |
| channel / author / music | partial | ✅ |
| view/like/comment stats | ❌ | ✅ |

---

## Order of execution
1. `youtube/capture.py` (fan-out, 8 procs) — fastest big win, gets chapters+transcript.
2. `transcribe.py` (GPU) — fills caption gaps.
3. `tiktok/capture.py` + whisper on TikTok clips.
4. `enrich.py` — join to clip level, emit `corpus.enriched.jsonl`.
5. Lift report.

Steps 1 and 3 are independent and can run concurrently.
