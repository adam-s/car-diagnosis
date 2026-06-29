"""Score the Qwen2-Audio second opinion against curated labels, alone and
JOINTLY with the CLAP head (creator-grouped OOF probs from confidence.py).

The question that matters: is agreement between two independent listeners a
stronger confidence signal than either alone, and does the audio-LLM rescue
the engine minority the CLAP head is confidently wrong about?

    uv run training/eval/eval_second_opinion.py
"""
import ast
import json
import sys
from collections import Counter

import numpy as np

from cardiag import paths

DATA = paths.TRAIN_DATA
from cardiag.training.models.confidence import feats, load_rows, oof_probs  # noqa: E402


def parse(text):
    try:
        blob = text[text.index("{"):text.rindex("}") + 1]
        try:
            o = json.loads(blob)
        except Exception:
            o = ast.literal_eval(blob)        # Qwen emits python-dict repr
        r = str(o.get("region", "")).lower()
        return {"region": r if r in ("engine", "chassis") else None,
                "sound": o.get("sound"), "part": o.get("part"),
                "conf": o.get("confidence")}
    except Exception:
        return {"region": None, "sound": None, "part": None, "conf": None}


def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else "smoke/qwen_audio.jsonl"
    man = {r["id"]: r for r in map(json.loads,
                                   open(DATA / "smoke/manifest.jsonl"))}
    qa = {r["id"]: parse(r["text"])
          for r in map(json.loads, open(DATA / qfile))}
    ok = {i for i in man if qa.get(i, {}).get("region")}
    dist = Counter(qa[i]["region"] for i in ok)
    print(f"{qfile}: {len(man)} clips, {len(ok)} parsed | "
          f"qwen region distribution {dict(dist)}")
    if len(dist) < 2:
        print("  -> COLLAPSED to a constant region; not discriminating.")

    # qwen alone
    for cls in ("engine", "chassis", None):
        ids = [i for i in ok if cls in (man[i]["label"], None)]
        corr = [qa[i]["region"] == man[i]["label"] for i in ids]
        print(f"  qwen-alone acc [{cls or 'ALL'}]: "
              f"{np.mean(corr):.3f} (n={len(ids)})")

    # joint with CLAP OOF (model never saw these creators)
    F = {"clap": feats()["clap"]}
    rows = load_rows(F)
    eids = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    cre = np.array([r[3] for r in rows])
    classes, P = oof_probs(np.array([F["clap"][e] for e in eids]),
                           y, cre, "logistic", "isotonic")
    clap = {e: (classes[np.argmax(p)], float(p.max()))
            for e, p in zip(eids, P)}

    both = [i for i in ok if i in clap]
    agree = [i for i in both if clap[i][0] == qa[i]["region"]]
    disag = [i for i in both if clap[i][0] != qa[i]["region"]]
    acc = lambda ids: (np.mean([clap[i][0] == man[i]["label"] for i in ids])
                       if ids else float("nan"))
    print(f"\njoint (n={len(both)}): CLAP-alone acc {acc(both):.3f}")
    print(f"  AGREE    n={len(agree):3d}  acc {acc(agree):.3f}")
    qacc = (np.mean([qa[i]['region'] == man[i]['label'] for i in disag])
            if disag else float('nan'))
    print(f"  DISAGREE n={len(disag):3d}  clap acc {acc(disag):.3f} | "
          f"qwen acc {qacc:.3f}")

    eng = [i for i in both if man[i]["label"] == "engine"]
    fixed = [i for i in eng if clap[i][0] != "engine"
             and qa[i]["region"] == "engine"]
    broke = [i for i in eng if clap[i][0] == "engine"
             and qa[i]["region"] != "engine"]
    print(f"\nengine minority (n={len(eng)}): clap wrong on "
          f"{sum(clap[i][0] != 'engine' for i in eng)}, "
          f"qwen rescues {len(fixed)}, qwen breaks {len(broke)}")
    print("\nsample qwen answers:")
    for i in list(ok)[:8]:
        print(f"  {man[i]['label']:7s}/{(man[i]['part'] or '')[:18]:18s} -> "
              f"qwen {qa[i]['region']:7s} {str(qa[i]['sound']):10s} "
              f"part={str(qa[i]['part'])[:24]} conf={qa[i]['conf']}")


if __name__ == "__main__":
    main()
