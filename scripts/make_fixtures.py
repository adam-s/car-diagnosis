"""Build the bundled offline fixtures (run once; the output is committed).

Scrapes a small real corpus, CLAP-embeds the clips, and saves a tiny
``embeddings.npz`` (embeddings + labels, NOT audio: license-clean and ~100 KB)
into the package. ``cardiag train --fixtures`` trains on this with no network and
no 2 GB CLAP download, so a fresh clone can produce a model in seconds.

    python scripts/make_fixtures.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# scrape into a scratch dir so we don't touch the user's data/
os.environ.setdefault(
    "CARDIAG_DATA",
    "/private/tmp/claude-501/-Users-adamsohn-Projects-car-diagnosis/"
    "3c9ee32b-a022-44ba-9830-4c91622f2921/scratchpad/fixture_build")

from cardiag import config  # noqa: E402
from cardiag.audio.clap import Clap  # noqa: E402
from cardiag.pipeline import build  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "src" / "cardiag" / "_fixtures"


def main():
    n = build.scrape_youtube(per_query=1, max_videos=16)
    print(f"scraped {n} clips; embedding…")
    rows = build.load_corpus()
    clap = Clap()
    X, kind, l1, cause, video, cid = [], [], [], [], [], []
    import librosa
    for r in rows:
        y, _ = librosa.load(r["wav"], sr=config.SR_CLAP, mono=True)
        if len(y) < config.SR_CLAP // 2:
            continue
        X.append(clap.embed([y])[0])
        kind.append(r.get("kind", ""))
        l1.append(r.get("l1", ""))
        cause.append(build._cause_of(r) or "")
        video.append(r.get("video", ""))
        cid.append(r["clip_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "embeddings.npz",
        X=np.array(X, dtype=np.float32),
        kind=np.array(kind), l1=np.array(l1), cause=np.array(cause),
        video=np.array(video), clip_id=np.array(cid))
    print(f"saved {len(X)} fixture embeddings -> {OUT/'embeddings.npz'}")
    print(f"  kind: {dict(zip(*np.unique(kind, return_counts=True)))}")
    print(f"  cause: {dict(zip(*np.unique(cause, return_counts=True)))}")


if __name__ == "__main__":
    main()
