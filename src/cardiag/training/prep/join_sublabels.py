"""Join text-mined faults down to clip-level SUB-LABELS.

YouTube: by TIMESTAMP (clip [start,end] overlaps a mined fault's time_range),
so a compilation video's clips each inherit the correct specific part.
TikTok: VIDEO-LEVEL (short single-fault clips; faults have no timestamps): all
clips of a video get its single named part.

Output (data/training/corpus_textmined_labels.jsonl), one row per matched clip:
    eid (embedding id: 'yo:'/'ti:'+clip_id), clip_id, video, platform,
    start, end, l1, tm_part, tm_category, tm_explicit, tm_trust

Trust: timestamp > video. Downstream trusts explicit+timestamp/video.

    uv run training/prep/join_sublabels.py
"""
import json
from collections import Counter

from cardiag import paths

DATA = paths.TRAIN_DATA
YT = paths.YT_DATA / "corpus.enriched.tiered.jsonl"
YT_RAW = paths.YT_DATA / "corpus.jsonl"   # new scrape clips
TT = paths.TT_DATA / "corpus_labeled.tiered.jsonl"


def valid_ranges(ranges):
    if not ranges:
        return
    # tolerate a flat single pair [a, b] (Qwen) as well as nested [[a, b], ...]
    if len(ranges) == 2 and all(isinstance(x, (int, float)) for x in ranges):
        ranges = [ranges]
    for r in ranges:
        if (isinstance(r, (list, tuple)) and len(r) == 2
                and all(isinstance(x, (int, float)) for x in r)
                and r[1] >= r[0]):
            yield r[0], r[1]


def overlaps(s, e, ranges):
    return any(not (e < a or s > b) for a, b in valid_ranges(ranges))


def main():
    mined = {}  # (platform, video) -> faults
    for l in open(DATA / "text_mined.jsonl"):
        r = json.loads(l)
        mined[(r.get("platform", "youtube"), r["video"])] = r["faults"]

    # YouTube clip records: enriched (has fused_kind) + raw scrape (new clips
    # from the overnight diagnostic scrape; no fusion yet, but their videos are
    # fault-diagnostic by construction, so timestamp overlap is the label).
    yt_records = []
    seen_clip = set()
    for l in open(YT):
        r = json.loads(l)
        yt_records.append(r)
        seen_clip.add(r["clip_id"])
    if YT_RAW.exists():
        for l in open(YT_RAW):
            r = json.loads(l)
            if r.get("file") and r["clip_id"] not in seen_clip:
                r["fused_kind"] = "fault"   # diagnostic-video clip
                yt_records.append(r)

    out = []
    n_seen = 0
    # ---- YouTube: timestamp join ----
    for r in yt_records:
        if r.get("fused_kind") != "fault":
            continue
        faults = mined.get(("youtube", r["video"]))
        if not faults:
            continue
        n_seen += 1
        s, e = r.get("start", 0), r.get("end", 0)
        hits = [f for f in faults if overlaps(s, e, f.get("time_ranges"))]
        chosen, trust = None, None
        if len(hits) == 1:
            chosen, trust = hits[0], "timestamp"
        elif len(hits) > 1:
            expl = [f for f in hits if f.get("explicit")]
            if len(expl) == 1:
                chosen, trust = expl[0], "timestamp"
        if chosen is None:
            parts = {f.get("part") for f in faults}
            if len(parts) == 1:
                chosen, trust = faults[0], "video"
        if chosen:
            # tightness: shortest matched range duration (smaller = the host
            # cued THIS sound; large = a whole-chapter span, coarser)
            durs = [b - a for a, b in valid_ranges(chosen.get("time_ranges"))
                    if not (e < a or s > b)]
            out.append({"eid": "yo:" + r["clip_id"], "clip_id": r["clip_id"],
                        "video": r["video"], "platform": "youtube",
                        "start": s, "end": e, "l1": r.get("l1"),
                        "tm_part": chosen.get("part"),
                        "tm_category": chosen.get("category"),
                        "tm_explicit": bool(chosen.get("explicit")),
                        "tm_trust": trust,
                        "tm_range_dur": round(min(durs), 1) if durs else None})

    # ---- TikTok: video-level join (single distinct part) ----
    for l in open(TT):
        r = json.loads(l)
        if r.get("fused_kind") != "fault":
            continue
        faults = mined.get(("tiktok", r["video"]))
        if not faults:
            continue
        n_seen += 1
        parts = {f.get("part") for f in faults}
        if len(parts) != 1:
            continue
        f = faults[0]
        out.append({"eid": "ti:" + r["clip_id"], "clip_id": r["clip_id"],
                    "video": r["video"], "platform": "tiktok",
                    "start": r.get("start"), "end": r.get("end"),
                    "l1": r.get("l1"), "tm_part": f.get("part"),
                    "tm_category": f.get("category"),
                    "tm_explicit": bool(f.get("explicit")),
                    "tm_trust": "video"})

    with open(DATA / "corpus_textmined_labels.jsonl", "w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")

    print(f"fault clips in mined videos: {n_seen} | sub-labeled: {len(out)}")
    print("platform:", dict(Counter(o["platform"] for o in out)))
    print("trust:", dict(Counter(o["tm_trust"] for o in out)))
    print("category:", dict(Counter(o["tm_category"] for o in out)
                            .most_common()))
    hi = [o for o in out if o["tm_explicit"]
          and (o["tm_trust"] == "timestamp" or o["platform"] == "tiktok")]
    print(f"\nhigh-trust (explicit; YT-timestamp or TT-video): {len(hi)} clips, "
          f"{len(set(o['video'] for o in hi))} videos")
    print("top sub-label parts (high-trust):")
    for k, v in Counter(o["tm_part"] for o in hi).most_common(25):
        print(f"  {v:4} {k}")


if __name__ == "__main__":
    main()
