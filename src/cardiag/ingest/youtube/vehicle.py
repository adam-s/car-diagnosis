"""Vehicle metadata enrichment (Haiku, opportunistic-only).

Per video: yt-dlp metadata fetch (title/description/upload_date, NO download),
then Haiku extracts year/make/model/mileage IF explicitly stated; null otherwise.
car_age_years = upload_year - model_year when both known. Normals-search videos
("2018 Camry cold start") are the richest source of YMM labels.

    uv run vehicle.py            # enriches all ledger videos -> data/vehicles.json
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from cardiag import config, paths

OUT = paths.YT_DATA / "vehicles.json"


def fetch_meta(vid):
    out = subprocess.run(
        ["yt-dlp", "--no-warnings", "--skip-download", "-j",
         f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, text=True, timeout=60).stdout
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return None
    return {"id": vid, "title": d.get("title", ""),
            "desc": (d.get("description") or "")[:300],
            "uploaded": (d.get("upload_date") or "")[:4]}


def haiku_extract(metas):
    prompt = (
        "For each YouTube car video, extract the vehicle's year, make, model, and "
        "mileage ONLY if explicitly stated in title/desc — null if not stated, never "
        "guess. If model year and upload year ('uploaded') are both known, set "
        "car_age_years = uploaded - year. Reply ONLY a JSON array of "
        '{"id":str,"year":int|null,"make":str|null,"model":str|null,'
        '"mileage_mi":int|null,"car_age_years":int|null}.\n' + json.dumps(metas))
    out = subprocess.run(["claude", "-p", "--model", config.HAIKU_MODEL, prompt],
                         capture_output=True, text=True, timeout=180).stdout
    try:
        return json.loads(out[out.index("["):out.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []


def main(vids=None, batch=20):
    if vids is None:
        vids = sorted({json.loads(l)["video"] for l in open(paths.YT_DATA / "corpus.jsonl")})
    known = json.loads(OUT.read_text()) if OUT.exists() else {}
    vids = [v for v in vids if v not in known]
    print(f"enriching {len(vids)} new videos (metadata only, no downloads)")
    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = [m for m in pool.map(fetch_meta, vids) if m]
    for i in range(0, len(metas), batch):
        for r in haiku_extract(metas[i:i + batch]):
            if r.get("id"):                 # guard: skip malformed LLM rows
                known[r["id"]] = r
        print(f"  {min(i+batch, len(metas))}/{len(metas)}")
    OUT.write_text(json.dumps(known, indent=2))

    have = [r for r in known.values() if r.get("make")]
    print(f"\n{len(known)} videos: make known {len(have)} "
          f"({100*len(have)/max(1,len(known)):.0f}%), "
          f"year {sum(1 for r in have if r.get('year'))}, "
          f"age {sum(1 for r in have if r.get('car_age_years') is not None)}, "
          f"mileage {sum(1 for r in have if r.get('mileage_mi'))}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
