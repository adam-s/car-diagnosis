"""The one embedding contract shared by training and inference.

Train and serve must turn audio into model-input vectors the **same** way, or the
linear heads see a different distribution at inference than they were fit on — the
classic *train/serve skew*. This module is the single place that produces a model
input, so the two paths cannot drift.

The contract has exactly one atomic unit:

  * :func:`embed_clip` — ONE audio span (a 1-D float array) -> ONE L2-normalized
    CLAP vector. This is the unit a head is **trained on** (one corpus clip) and
    **scored on** (one isolated span at inference). Both call this exact function.

A *recording* (an uploaded clip) usually contains several spans. Inference turns
it into the list of per-span vectors a head should score:

  * :func:`model_vectors` — a recording -> an (n_spans, dim) array, each row an
    :func:`embed_clip` of one span, plus the spans / clean-result for reporting.

Aggregation over those spans happens in **probability space** (mean of each
head's ``predict_proba``), never by averaging the vectors. Averaging
L2-normalized embeddings and renormalizing produces a vector the StandardScaler
and LogisticRegression never saw at fit time — exactly the skew this module
exists to prevent. Keeping every vector a single-span embedding makes train and
serve identical at the level that matters: what reaches the classifier.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cardiag import config
from cardiag.audio.clap import Clap
from cardiag.audio.clean import clean


def embed_clip(y: np.ndarray, sr: int = config.SR_CLAP) -> np.ndarray:
    """One audio span -> one L2-normalized CLAP vector.

    The atomic unit of the whole system: training embeds each corpus clip with
    this, and inference embeds each isolated span with this. Same function, same
    distribution — no skew possible.
    """
    return Clap().embed([y], sr=sr)[0]


def embed_clips(ys, sr: int = config.SR_CLAP) -> np.ndarray:
    """:func:`embed_clip` over many spans -> (n, dim). Each row is independent."""
    if not len(ys):
        return np.zeros((0, 0), dtype=np.float32)
    return Clap().embed(list(ys), sr=sr)


@dataclass
class EmbedResult:
    """Every model-input vector for one recording, plus reporting context.

    ``vectors`` is (n_spans, dim); each row is a single-span embedding — never an
    average — so it is in-distribution for the heads. The caller pools the heads'
    probabilities across the rows.
    """
    vectors: np.ndarray
    segments: list = field(default_factory=list)
    clean_result: object | None = None
    source: str = "windows"          # "isolated" | "windows"

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0]) if self.vectors.size else 0


def _window_vectors(path, win_s: float = 10.0, sr: int = config.SR_CLAP) -> np.ndarray:
    """Embed up to 3 windows of a whole file, each via :func:`embed_clip`.

    The fallback when cleaning isolates no span (e.g. a phone clip that is all
    one sound): rather than average the windows into a single vector, we return
    one vector per window so the caller can pool their probabilities — the same
    span-as-a-sample treatment training uses.
    """
    import librosa
    if not Path(path).exists():
        raise FileNotFoundError(f"no such audio file: {path}")
    try:                                    # probe readability BEFORE loading CLAP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dur = librosa.get_duration(path=str(path))
    except Exception as e:
        raise ValueError(f"could not read audio from {path} — is it a valid "
                         f"audio file? ({type(e).__name__})") from None
    offs = [0.0] if dur <= win_s else [0.0, (dur - win_s) / 2, dur - win_s][:3]
    vecs = []
    for off in offs:
        y, _ = librosa.load(str(path), sr=sr, mono=True, offset=max(0.0, off),
                            duration=win_s)
        if len(y) < sr // 2:
            continue
        vecs.append(embed_clip(y, sr=sr))
    if not vecs:
        raise ValueError(f"no usable audio in {path}")
    return np.array(vecs)


def model_vectors(path, *, clean_audio: bool = True,
                  win_s: float = 10.0) -> EmbedResult:
    """A recording -> the per-span vectors its heads should score.

    With ``clean_audio`` (the default), run the same cleaning cascade the corpus
    was built with and embed each isolated mechanical span. If cleaning isolates
    nothing, fall back to windows of the whole file (still one vector per window).
    With ``clean_audio=False``, skip cleaning and embed windows directly — used
    when the input is already an isolated clip (a corpus clip, or a test tone).
    """
    if clean_audio:
        res = clean(path)
        if res.isolated:
            X = embed_clips(res.isolated, sr=res.sr)
            return EmbedResult(X, res.segments, res, "isolated")
        # cascade isolated nothing — diagnose the whole clip, keep res for notes
        return EmbedResult(_window_vectors(path, win_s), res.segments, res, "windows")
    return EmbedResult(_window_vectors(path, win_s), [], None, "windows")
