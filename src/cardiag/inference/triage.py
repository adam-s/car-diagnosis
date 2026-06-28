"""``TriageClassifier`` — the honest headline product.

Audio reliably separates exactly one distinction: **engine-internal** noise vs
**running-gear** (wheel / suspension / driveline) noise. This returns that call
plus a *calibrated* probability and a HIGH / MEDIUM / LOW / ABSTAIN band, so a
human knows when to trust it (validated held-out: cov@p80 1.0, cov@p90 0.82).

It deliberately does NOT name the exact part — audio can't, and the model stays
honest by abstaining instead of guessing.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from cardiag import paths
from cardiag.audio.clap import embed_windows
from cardiag.types import Band, TriageResult

_PLAIN = {
    "engine": "ENGINE-INTERNAL — from inside the running engine "
              "(knock, lifter/valvetrain tick, low-oil rattle).",
    "chassis": "RUNNING GEAR — wheels / suspension / driveline "
               "(wheel-bearing hum, CV-joint click, suspension clunk).",
}
_NEXT = {
    "engine": "Check oil level/condition; note if it tracks engine RPM. "
              "Treat internal engine noise as urgent.",
    "chassis": "Note when it happens — over bumps, while turning, or rising "
               "with road speed — to localize wheel vs suspension vs driveline.",
}


def _band(conf: float) -> tuple[Band, str]:
    # thresholds + glosses from the measured held-out reliability table
    # (creator-grouped OOF: HIGH 90% / MEDIUM 84% / LOW 82% / abstain 57%)
    if conf >= 0.90:
        return Band.HIGH, "~90% of similar held-out cases were correct"
    if conf >= 0.80:
        return Band.MEDIUM, "~84% of similar held-out cases were correct"
    if conf >= 0.65:
        return Band.LOW, "a lean, not a call — verify"
    return Band.ABSTAIN, "too close to call from this audio"


class TriageClassifier:
    """Calibrated engine-vs-running-gear triage. Construct via :meth:`load`."""

    def __init__(self, model, classes):
        self.model = model
        self.classes = np.array(classes)

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> TriageClassifier:
        path = Path(model_path or paths.MODEL_TRIAGE)
        if not path.exists():
            raise FileNotFoundError(
                f"No triage model at {path}.\n"
                f"  → quickest fix (offline, ~2s):  cardiag train --fixtures\n"
                f"  → or point --model / CARDIAG_DATA at an existing "
                f"triage_model.joblib."
            )
        art = joblib.load(path)
        return cls(art["model"], art["classes"])

    def triage(self, path) -> TriageResult:
        x = embed_windows(path)
        p = self.model.predict_proba([x])[0]
        i = int(p.argmax())
        label = str(self.classes[i])
        conf = float(p[i])
        band, gloss = _band(conf)
        return TriageResult(
            file=str(path),
            triage=label,
            label=_PLAIN.get(label, label),
            confidence=conf,
            band=band,
            band_gloss=gloss,
            probabilities={str(c): float(v) for c, v in zip(self.classes, p)},
            next_step="" if band is Band.ABSTAIN else _NEXT.get(label, ""),
        )
