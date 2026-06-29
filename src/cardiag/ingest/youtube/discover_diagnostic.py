"""Discover sub-type DIAGNOSTIC videos: the tight-timestamp label source.

The overnight finding: tight, host-cued labels ("here's what a bad X sounds
like") are the lever. Those come from narrated diagnostic videos, not
compilations. This searches sub-type-specific "what does X sound like" queries
(deep), dedups against the existing worklist + ledger + already-captured meta,
and MERGES new videos into data/youtube/worklist.json (never clobbers).

Allows longer durations than the compilation-tuned config (narrated diagnostics
run 1-15 min and carry the timestamped labels).

    uv run ingest/youtube/discover_diagnostic.py [per_query=80]
"""
import json
import subprocess
import sys

from cardiag import paths

WORKLIST = paths.YT_DATA / "worklist.json"
LEDGER = paths.YT_DATA / "corpus.jsonl"
META = paths.YT_DATA / "meta"
DUR_MIN, DUR_MAX = 45, 1200   # narrated diagnostics, not 25-min vlogs

QUERIES = [
    # engine internals (the subtype targets)
    "what does a bad rod bearing sound like", "rod knock sound diagnosis",
    "engine knocking sound diagnosis", "lifter tick sound diagnosis",
    "valve lifter noise sound", "rocker arm noise sound",
    "timing chain rattle sound", "piston slap sound cold start",
    "spark knock detonation sound", "low oil engine noise sound",
    "main bearing knock sound", "wrist pin knock sound",
    # rotating / driveline
    "what does a bad wheel bearing sound like", "wheel bearing noise diagnosis",
    "bad cv joint sound clicking", "cv axle noise diagnosis",
    "bad u joint sound", "carrier bearing noise sound",
    "differential whine noise sound", "rear differential noise sound",
    "transmission whine sound", "wheel bearing vs cv joint sound",
    # accessory / belt
    "serpentine belt noise sound", "belt tensioner noise sound",
    "bad idler pulley sound", "bad alternator sound whine",
    "power steering pump whine sound", "water pump noise sound bearing",
    "ac compressor noise sound", "bad pulley bearing sound",
    # suspension / steering / brakes
    "bad ball joint sound clunk", "tie rod end noise sound",
    "sway bar link noise clunk", "bad strut noise sound",
    "control arm bushing noise", "clunking noise over bumps diagnosis",
    "brake grinding noise sound", "exhaust leak sound diagnosis",
    "vacuum leak sound diagnosis", "heat shield rattle sound",
    # the high-value "demonstration" videos
    "guess the car noise", "car noises and what they mean explained",
    "diagnosing car noises by sound", "common car noises explained mechanic",
]


def search(q, n):
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--flat-playlist",
             "--print", "%(id)s\t%(duration)s\t%(title)s", f"ytsearch{n}:{q}"],
            capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0]:
            try:
                d = float(p[1])
            except ValueError:
                continue
            if DUR_MIN <= d <= DUR_MAX:
                rows.append({"id": p[0], "dur": d, "title": p[2],
                             "query": q, "kind": "fault"})
    return rows


def main(per_query=80):
    work = json.load(open(WORKLIST)) if WORKLIST.exists() else []
    have = {w["id"] for w in work}
    if LEDGER.exists():
        have |= {json.loads(l)["video"] for l in open(LEDGER)}
    added = 0
    for i, q in enumerate(QUERIES, 1):
        for r in search(q, per_query):
            if r["id"] not in have:
                have.add(r["id"])
                work.append(r)
                added += 1
        print(f"[{i}/{len(QUERIES)}] {q[:46]:<46} +new total {added}",
              flush=True)
    WORKLIST.write_text(json.dumps(work, indent=2))
    print(f"\nadded {added} new diagnostic videos -> {len(work)} in worklist")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
