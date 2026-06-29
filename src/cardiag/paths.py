"""Central path resolution. Every module that touches disk imports from here
instead of computing ``parent.parent`` and hardcoding directory names.

The data tree is large and regenerable, so it is gitignored and lives outside
the importable package:

    <repo>/data/
      youtube/  tiktok/  reddit/        corpora + clips, per platform
      external/{car-diagnostics,ai-mechanic,sound-based-db1,car-engine}
      training/                         manifests, *.npz caches, *.joblib models

Override the location with the ``CARDIAG_DATA`` environment variable (useful for
tests and CI, which point it at a fixture dir).

Historical clip paths stored in the corpora use the *old* repo layout
(e.g. ``youtube/data/clips/x.wav``); they are treated as stable logical ids and
translated to their on-disk location by :func:`resolve_clip`, so the data never
has to be rewritten and id-keyed embedding caches stay valid.
"""
from __future__ import annotations

import os
from pathlib import Path

# Package dir is <repo>/src/cardiag/ ; walk up to the repo root (the dir that
# contains pyproject.toml). Falls back to two levels up if not found.
_PKG = Path(__file__).resolve().parent
REPO_ROOT = next(
    (p for p in _PKG.parents if (p / "pyproject.toml").exists()),
    _PKG.parent.parent,
)

# --- data roots ------------------------------------------------------------
DATA = Path(os.environ.get("CARDIAG_DATA", REPO_ROOT / "data")).resolve()
YT_DATA = DATA / "youtube"
TT_DATA = DATA / "tiktok"
REDDIT_DATA = DATA / "reddit"
TRAIN_DATA = DATA / "training"
EXTERNAL = DATA / "external"

# --- bundled model artifacts (under data/training) -------------------------
MODEL_CLAP = TRAIN_DATA / "best_model_clap.joblib"   # kind / knock / cause heads
MODEL_TRIAGE = TRAIN_DATA / "triage_model.joblib"    # engine vs running-gear

# A tiny synthetic demo clip (a generated engine knock, no copyright) so a fresh
# clone can `diagnose` something without scraping anything.
DEMO_CLIP = _PKG / "_fixtures" / "demo.wav"

# The optional pre-trained model that ships with the repo. If the user hasn't
# trained their own (data/training/ is empty), we fall back to this so a fresh
# clone can `diagnose` / `serve` immediately. An explicit --model always wins.
SHIPPED_DIR = REPO_ROOT / "models"


def resolve_clap() -> Path:
    """The fault/knock/cause model to load: a user-trained one if present, else the
    shipped one, else the (missing) default path for a clean error message."""
    if MODEL_CLAP.exists():
        return MODEL_CLAP
    shipped = SHIPPED_DIR / "best_model_clap.joblib"
    return shipped if shipped.exists() else MODEL_CLAP


def resolve_triage() -> Path:
    if MODEL_TRIAGE.exists():
        return MODEL_TRIAGE
    shipped = SHIPPED_DIR / "triage_model.joblib"
    return shipped if shipped.exists() else MODEL_TRIAGE

# --- external reference dataset roots (consumed by training/prep) ----------
CARDIAG_DS = EXTERNAL / "car-diagnostics" / "car diagnostics dataset"
AIMECH = EXTERNAL / "ai-mechanic" / "ML dataset" / "Audio"
DB1 = EXTERNAL / "sound-based-db1" / "Datasets" / "DB1"
CARENGINE = EXTERNAL / "car-engine" / "_audio"

# --- clip path translation -------------------------------------------------
# Stored paths use the OLD layout; map their prefixes to the NEW layout.
_PREFIX_MAP = {
    "youtube/data/": "youtube/",
    "tiktok/data/": "tiktok/",
    "reddit/data/": "reddit/",
    "training1/data/": "training/",
    "external-data/kaggle/car-diagnostics-dataset/": "external/car-diagnostics/",
    "external-data/repos/Ai_Mechanic/": "external/ai-mechanic/",
    "external-data/repos/Sound-Based-Vehicle-Diagnostics-Emergency-Signal-Recognition/":
        "external/sound-based-db1/",
    "external-data/repos/Car-Engine-Sounds-Dataset/": "external/car-engine/",
}
_PREFIXES = sorted(_PREFIX_MAP, key=len, reverse=True)


def resolve_clip(path) -> Path:
    """Map a stored (old-layout, repo-relative) clip path to its absolute
    location under :data:`DATA`. Absolute paths pass through unchanged."""
    s = str(path)
    p = Path(s)
    if p.is_absolute():
        return p
    for pre in _PREFIXES:
        if s.startswith(pre):
            return DATA / (_PREFIX_MAP[pre] + s[len(pre):])
    return DATA / s


def ensure_data_dirs() -> None:
    """Create the per-platform data roots (idempotent)."""
    for d in (YT_DATA, TT_DATA, REDDIT_DATA, TRAIN_DATA, EXTERNAL):
        d.mkdir(parents=True, exist_ok=True)
