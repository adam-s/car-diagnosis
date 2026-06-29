"""Discovery: harvest a deduped worklist from fault + normal query sets.

Each entry records which query set found it (kind: fault|normal); normals are
the negative/baseline class and tend to carry year/make/model in their titles.
Lesson encoded: prefer 60-180s compilations (purest labels); skip >25min vlogs.

    uv run discover.py [per_query]
"""
import json
import subprocess
import sys

from cardiag import config, paths


def harvest(query, n):
    out = subprocess.run(
        ["yt-dlp", "--no-warnings", "--flat-playlist",
         "--print", "%(id)s\t%(duration)s\t%(title)s",
         f"ytsearch{n}:{query}"],
        capture_output=True, text=True, timeout=120).stdout
    rows = []
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0]:
            try:
                d = float(p[1])
            except ValueError:
                continue
            if config.DUR_MIN_S <= d <= config.DUR_MAX_S:
                rows.append({"id": p[0], "dur": d, "title": p[2], "query": query})
    return rows


def main(per_query=40):
    seen, work = set(), []
    for kind, queries in (("fault", config.FAULT_QUERIES),
                          ("normal", config.NORMAL_QUERIES)):
        for q in queries:
            for r in harvest(q, per_query):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    work.append({**r, "kind": kind})
            print(f"[{kind}] {q:<44} unique so far: {len(work)}")
    paths.YT_DATA.mkdir(exist_ok=True)
    (paths.YT_DATA / "worklist.json").write_text(json.dumps(work, indent=2))
    nf = sum(w["kind"] == "fault" for w in work)
    print(f"\n{len(work)} videos -> data/worklist.json ({nf} fault, {len(work)-nf} normal)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
