"""Haiku-normalize YouTube chapter labels -> clean cause labels + content kind.

enrich.py attaches raw chapter_label to 32% of clips, but crude keyword matching
mapped only 11%: chapters phrased "Loose Pinion Bearings?" or "Whining When
Cold" are obvious to a reader, opaque to keywords. And the corpus is polluted
with enthusiast car-sound videos (Ferrari/Porsche revving = healthy NORMAL
sounds, not faults) and junk chapters (Intro, music tracks).

Haiku maps each UNIQUE chapter string once (dedup -> cheap) to:
  - part : canonical fault part, or null
  - kind : "fault" | "normal" (healthy/enthusiast engine note) | "nonauto" (junk)

Writes corpus.enriched.jsonl in place with l2_chapter + kind on every clip.

    uv run normalize.py
"""
import json
import subprocess
from collections import Counter

from cardiag import config, paths

SRC = paths.YT_DATA / "corpus.enriched.jsonl"


def haiku_map(labels, batch=40):
    out = {}
    items = [{"i": i, "label": l[:80]} for i, l in enumerate(labels)]
    for k in range(0, len(items), batch):
        chunk = items[k:k + batch]
        prompt = (
            "Each item is a YouTube video CHAPTER TITLE from car-sound videos. For "
            "each: (1) part = the canonical automotive fault part it names (lowercase, "
            "e.g. 'wheel bearing','valve cover gasket'), or null if it names no specific "
            "part; (2) kind = 'fault' if it describes a problem/noise, 'normal' if it's a "
            "healthy engine/exhaust note or enthusiast car showcase (e.g. a Ferrari rev, "
            "'cold start sound'), or 'nonauto' if it's junk (Intro, Outro, a music track, "
            "a car model name with no fault). Reply ONLY a JSON array of "
            '{"i":int,"part":str|null,"kind":str}.\n' + json.dumps(chunk))
        try:
            r = subprocess.run(["claude", "-p", "--model", config.HAIKU_MODEL, prompt],
                               capture_output=True, text=True, timeout=150).stdout
            for o in json.loads(r[r.index("["):r.rindex("]") + 1]):
                out[o["i"]] = (o.get("part"), o.get("kind", "fault"))
        except (ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        print(f"  mapped {min(k+batch, len(items))}/{len(items)} unique chapters", flush=True)
    return out


def main():
    recs = [json.loads(l) for l in open(SRC)]
    uniq = sorted({r["chapter_label"] for r in recs if r.get("chapter_label")})
    print(f"{len(recs)} clips, {len(uniq)} unique chapter strings -> Haiku")
    mp = haiku_map(uniq)
    lookup = {uniq[i]: v for i, v in mp.items()}

    tiers = Counter()
    for r in recs:
        part, kind = lookup.get(r.get("chapter_label"), (None, None))
        r["l2_chapter"] = part
        if kind:
            r["kind"] = kind            # fault | normal | nonauto (chapter-derived)
        tiers[kind or "no-chapter"] += 1
    open(SRC, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")

    withpart = sum(1 for r in recs if r.get("l2_chapter"))
    print(f"\nclips with a Haiku chapter-part: {withpart} "
          f"({100*withpart/len(recs):.0f}%)  [keyword version was 11%]")
    print(f"content kind: {dict(tiers)}")
    print(f"chapter parts: {dict(Counter(r['l2_chapter'] for r in recs if r.get('l2_chapter')).most_common(15))}")


if __name__ == "__main__":
    main()
