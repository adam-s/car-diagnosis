"""Mine TikTok sub-labels from query + desc + OCR (no transcript on TikTok).

TikTok clips carry the SEARCH QUERY that found them ("bad wheel bearing sound"),
the caption (desc), and overlay OCR. Videos are short and usually a single
fault, so Haiku extracts one (or few) specific part(s) per video, adding
distinct videos to the sub-label pool, which is the binding constraint for the
fine sub-type CV.

Writes to the same text_mined.jsonl with platform="tiktok" and time_ranges=[]
(short clips -> video-level join downstream).

    uv run training/prep/text_mine_tt.py --limit 900
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from cardiag import paths

DATA = paths.TRAIN_DATA
CACHE = DATA / "text_mined.jsonl"
TT = paths.TT_DATA / "corpus_labeled.tiered.jsonl"
MODEL = "claude-haiku-4-5"
WORKERS = 6

INSTR = (
    "You are an expert mechanic. A short TikTok car-noise clip has this text: a "
    "SEARCH QUERY it was found under, the caption (desc), and on-screen overlay "
    "text (ocr, often junk/usernames — ignore unless it names a part). Identify "
    "the specific mechanical fault(s) shown. Be SPECIFIC about engine noises "
    "(rod/bearing knock, lifter/valvetrain tick, low-oil, detonation, timing "
    "chain). For each fault return: part (lowercase specific component), "
    "category (engine_internal|valvetrain|low_oil|fuel_ignition|suspension|"
    "driveline|brakes|exhaust|accessory_belt|steering|cooling|other), explicit "
    "(true if the query or caption NAMES it, false if guessed), evidence (short "
    "quote). Return ONLY raw JSON: {\"faults\":[{...}]}. No prose."
)


def tt_videos(limit):
    by_vid = {}
    for l in open(TT):
        r = json.loads(l)
        if r.get("fused_kind") != "fault":
            continue
        v = r["video"]
        d = by_vid.setdefault(v, {"queries": set(), "descs": set(),
                                  "ocr": set()})
        if r.get("query"):
            d["queries"].add(r["query"])
        if r.get("desc"):
            d["descs"].add(r["desc"][:200])
        ocr = r.get("ocr_label")
        if ocr and len(ocr) > 3:
            d["ocr"].add(ocr[:40])
    return list(by_vid.items())[:limit]


def mine_one(vid, info):
    bundle = {"query": sorted(info["queries"]),
              "desc": sorted(info["descs"])[:3],
              "ocr": sorted(info["ocr"])[:8]}
    prompt = INSTR + "\n\nCLIP TEXT:\n" + json.dumps(bundle)[:4000]
    try:
        out = subprocess.run(["claude", "-p", "--model", MODEL, prompt],
                             capture_output=True, text=True,
                             timeout=120).stdout
        obj = json.loads(out[out.index("{"):out.rindex("}") + 1])
        faults = obj.get("faults", [])
        for f in faults:
            f["time_ranges"] = []
        return vid, faults
    except Exception:
        return vid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=900)
    args = ap.parse_args()
    vids = tt_videos(args.limit)
    cached = set()
    if CACHE.exists():
        cached = {json.loads(l)["video"] for l in open(CACHE)}
    vids = [(v, i) for v, i in vids if v not in cached]
    print(f"mining {len(vids)} TikTok videos with {MODEL}", flush=True)
    caf = open(CACHE, "a")
    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(mine_one, v, i) for v, i in vids]
        for fut in as_completed(futs):
            vid, faults = fut.result()
            if faults is None:
                continue
            caf.write(json.dumps({"video": vid, "platform": "tiktok",
                                  "faults": faults}) + "\n")
            caf.flush()
            ok += 1
            if ok % 25 == 0:
                print(f"  {ok}/{len(vids)} mined", flush=True)
    caf.close()
    print(f"done: {ok} TikTok videos mined")


if __name__ == "__main__":
    main()
