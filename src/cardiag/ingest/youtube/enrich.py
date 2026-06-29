"""Join captured metadata down to the clip level (the payoff of capture.py).

For every clip in corpus.jsonl, attach from its video's data/meta/<id>/:
  - chapter_label   : title of the chapter containing the clip midpoint
                      (uploader-authored, timestamped fault label, near GT)
  - clip_transcript : caption cues overlapping the clip window (local narration)
  - l2_local        : fault category from chapter_label + clip_transcript,
                      REPLACING the weak global-title l2_candidates
  - video tags / description / top_comments for downstream use

Emits corpus.enriched.jsonl and a lift report (how many clips gained a local
label, how many l2_local disagree with the old title-derived l2).

    uv run enrich.py
"""
import glob
import json
from collections import Counter

from cardiag import config, paths

META = paths.YT_DATA / "meta"
PAD = 1.0   # seconds of slack when overlapping transcript cues with a clip


def load_chapters(info):
    return [(c["start_time"], c["end_time"], c["title"]) for c in (info.get("chapters") or [])]


def load_captions(vid):
    """Best English json3 track -> [(start_s, end_s, text)]."""
    files = sorted(glob.glob(str(META / vid / f"{vid}*.json3")))
    if not files:
        return []
    # prefer manual en over auto (orig); json3 events have tStartMs + segs
    pick = next((f for f in files if ".en." in f and "orig" not in f), files[0])
    try:
        ev = json.load(open(pick)).get("events") or []
    except (json.JSONDecodeError, OSError):
        return []
    cues = []
    for e in ev:
        segs = e.get("segs") or []
        txt = "".join(s.get("utf8", "") for s in segs).strip()
        if txt and "tStartMs" in e:
            s = e["tStartMs"] / 1000.0
            cues.append((s, s + e.get("dDurationMs", 2000) / 1000.0, txt))
    return cues


def l2_from_text(text):
    tl = (text or "").lower()
    return sorted({p for p, kws in config.L2_KEYWORDS.items() if any(k in tl for k in kws)})


def main():
    recs = [json.loads(l) for l in open(paths.YT_DATA / "corpus.jsonl")]
    # cache per-video metadata
    cache = {}

    def meta(vid):
        if vid not in cache:
            ij = META / vid / f"{vid}.info.json"
            info = json.loads(ij.read_text()) if ij.exists() else {}
            cache[vid] = {
                "chapters": load_chapters(info),
                "captions": load_captions(vid),
                "tags": (info.get("tags") or [])[:20],
                "description": (info.get("description") or "")[:1000],
                "comments": [c.get("text", "")[:200] for c in (info.get("comments") or [])[:10]],
                "has_meta": bool(info),
            }
        return cache[vid]

    out = open(paths.YT_DATA / "corpus.enriched.jsonl", "w")
    n_chap = n_tx = n_local = n_changed = n_meta = 0
    for r in recs:
        m = meta(r["video"])
        n_meta += 1 if m["has_meta"] else 0
        mid = (r["start"] + r["end"]) / 2
        chap = next((t for s, e, t in m["chapters"] if s <= mid < e), None)
        tx = [t for s, e, t in m["captions"]
              if not (e < r["start"] - PAD or s > r["end"] + PAD)]
        # local cause: chapter title first (clean), else local transcript
        l2_local = l2_from_text(chap) or l2_from_text(" ".join(tx))
        if chap:
            n_chap += 1
        if tx:
            n_tx += 1
        if l2_local:
            n_local += 1
        if set(l2_local) != set(r.get("l2_candidates") or []):
            n_changed += 1
        r["chapter_label"] = chap
        r["clip_transcript"] = " ".join(tx)[:500]
        r["l2_local"] = l2_local
        r["video_tags"] = m["tags"]
        r["top_comments"] = m["comments"]
        out.write(json.dumps(r) + "\n")
    out.close()

    N = len(recs)
    print(f"enriched {N} clips ({n_meta} had metadata)")
    print(f"  chapter_label:   {n_chap} ({100*n_chap/N:.0f}%)")
    print(f"  clip_transcript: {n_tx} ({100*n_tx/N:.0f}%)")
    print(f"  l2_local set:    {n_local} ({100*n_local/N:.0f}%)  vs old title-l2")
    print(f"  l2 CHANGED from title-derived: {n_changed} ({100*n_changed/N:.0f}%)")
    locals_ = Counter(p for r in (json.loads(l) for l in open(paths.YT_DATA / 'corpus.enriched.jsonl'))
                      for p in r["l2_local"])
    print(f"  l2_local distribution: {dict(locals_.most_common(15))}")


if __name__ == "__main__":
    main()
