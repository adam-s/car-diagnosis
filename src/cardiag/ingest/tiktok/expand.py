"""Catalog expansion: turn each on-topic creator from the search worklist into
their whole back-catalog. yt-dlp CAN enumerate a TikTok account page (unlike
search), so this needs no browser.

A creator who showed up for >=2 distinct fault queries is almost certainly a
car-content channel -> their catalog is dense with labeled montages. We also
seed a few hand-verified accounts. Catalog videos are kept only if short
(montage range) and car-keyworded; the OCR/CLAP labeler filters the rest.

    python -m cardiag.ingest.tiktok.expand
"""
import json
import subprocess
from collections import Counter

from cardiag import paths

DATA = paths.TT_DATA
SEED_ACCOUNTS = ["accurateautoinc", "car.tipx", "motogearlab8", "endjdiy25"]
CAR_KW = ("car", "noise", "sound", "engine", "brake", "bearing", "mechanic",
          "suspension", "wheel", "belt", "joint", "knock", "squeal", "rattle",
          "tick", "whine", "grind", "clunk", "auto", "vehicle", "truck", "fix")
DUR_MIN, DUR_MAX = 5, 90
PER_ACCOUNT = 60


def catalog(handle):
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-warnings", "--flat-playlist", "--playlist-end", str(PER_ACCOUNT),
             "--print", "%(id)s\t%(duration)s\t%(title).80s",
             f"https://www.tiktok.com/@{handle}"],
            capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return []
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) < 3 or not p[0]:
            continue
        try:
            d = float(p[1])
        except ValueError:
            continue
        title = p[2]
        if DUR_MIN <= d <= DUR_MAX and any(k in title.lower() for k in CAR_KW):
            rows.append({"id": p[0], "author": handle, "desc": title,
                         "url": f"https://www.tiktok.com/@{handle}/video/{p[0]}",
                         "platform": "tiktok", "query": f"catalog:@{handle}"})
    return rows


def main():
    wl = [json.loads(l) for l in open(paths.TT_DATA / "worklist.jsonl")]
    seen = {w["id"] for w in wl}
    by_author = Counter(w["author"] for w in wl)
    # creators with >=2 hits across the fault searches + hand-seeded ones
    accounts = sorted({a for a, n in by_author.items() if n >= 2} | set(SEED_ACCOUNTS))
    print(f"expanding {len(accounts)} on-topic accounts")

    new = 0
    with open(paths.TT_DATA / "worklist.jsonl", "a") as f:
        for h in accounts:
            rows = catalog(h)
            added = 0
            for r in rows:
                if r["id"] not in seen:
                    f.write(json.dumps(r) + "\n"); seen.add(r["id"]); new += 1; added += 1
            print(f"  @{h:<22} +{added} (catalog {len(rows)})", flush=True)
    print(f"\n+{new} from catalogs; worklist now {len(seen)}")


if __name__ == "__main__":
    main()
