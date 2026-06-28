"""``Classifier`` — turn one recording into a structured :class:`Diagnosis`.

Loads the clean-teacher heads (trained by ``training/models/train_best.py``)
over frozen CLAP embeddings: ``kind`` (fault vs normal), ``knock`` (engine knock
vs normal), and ``cause`` (part family). Honest by design — cause is top-k with
probabilities, never a single confident answer (fine cause from audio alone has
a measured ceiling).

    from cardiag import Classifier
    clf = Classifier.load()
    print(clf.diagnose("clip.wav").to_dict())
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from cardiag import paths
from cardiag.audio.clap import Clap, embed_windows
from cardiag.audio.clean import clean
from cardiag.types import Cause, Diagnosis, Verdict

CAUSE_HELP = {
    "brakes": "worn brake pads/rotors — inspect brakes",
    "belt": "serpentine/accessory belt or tensioner",
    "power_steering": "power-steering pump or fluid",
    "accessories": "alternator / starter / battery / A-C",
    "fuel_ignition": "ignition or fuel delivery (hard start)",
    "low_oil": "low oil / top-end — check oil level NOW",
}

_FAULT_HI, _FAULT_LO = 0.6, 0.4


def _proba(clf, x) -> dict:
    return dict(zip(clf.classes_, clf.predict_proba([x])[0]))


class Classifier:
    """Bundle of the three trained heads. Construct via :meth:`load`."""

    def __init__(self, heads: dict):
        self.heads = heads

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> "Classifier":
        """Load heads from a joblib artifact (defaults to the bundled model)."""
        path = Path(model_path or paths.MODEL_CLAP)
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}. Train one with `cardiag train`, or set "
                f"--model / CARDIAG_DATA to point at an existing "
                f"best_model_clap.joblib."
            )
        return cls(joblib.load(path)["heads"])

    def _embed(self, path, *, clean_audio: bool):
        """Clean the clip then embed the isolated mechanical audio. Falls back to
        windowed embedding of the whole file when cleaning isolates nothing."""
        segments = []
        if clean_audio:
            res = clean(path)
            segments = res.segments
            if res.isolated:
                vecs = Clap().embed(res.isolated, sr=res.sr)
                v = vecs.mean(0)
                return v / (np.linalg.norm(v) + 1e-9), segments, res
            res_for_note = res
        else:
            res_for_note = None
        return embed_windows(path), segments, res_for_note

    def diagnose(self, path, *, clean_audio: bool = True) -> Diagnosis:
        """Diagnose ``path``: clean -> embed -> heads -> :class:`Diagnosis`."""
        x, segments, res = self._embed(path, clean_audio=clean_audio)

        p_fault = float(_proba(self.heads["kind"], x).get("fault", 0.0))
        p_knock = float(_proba(self.heads["knock"], x).get("knock", 0.0))
        cause = _proba(self.heads["cause"], x)
        topk = sorted(cause.items(), key=lambda kv: -kv[1])[:3]

        verdict = (Verdict.FAULT if p_fault >= _FAULT_HI else
                   Verdict.NORMAL if p_fault <= _FAULT_LO else Verdict.UNCERTAIN)
        causes = [Cause(part=k, p=round(float(v), 3), note=CAUSE_HELP.get(k, ""))
                  for k, v in topk]

        note = Diagnosis.note
        if res is not None and getattr(res, "is_music", False):
            note = "Recording looks like mostly music — diagnosis is unreliable. " + note
        elif res is not None and getattr(res, "is_empty", False):
            note = ("Cleaning isolated no clear mechanical sound; diagnosed the "
                    "whole clip. " + note)

        return Diagnosis(
            file=str(path),
            verdict=verdict,
            fault_probability=p_fault,
            engine_knock_probability=p_knock,
            causes=causes,
            segments=segments,
            note=note,
        )
