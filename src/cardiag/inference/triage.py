"""``TriageClassifier``: the honest headline product.

Audio reliably separates exactly one distinction: **engine-internal** noise vs
**running-gear** (wheel / suspension / driveline) noise. This returns that call
plus a *calibrated* probability and a HIGH / MEDIUM / LOW / ABSTAIN band, so a
human knows when to trust it (validated held-out: cov@p80 1.0, cov@p90 0.82).

It deliberately does NOT name the exact part: audio can't, and the model stays
honest by abstaining instead of guessing.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from cardiag import paths
from cardiag.audio.embed import model_vectors
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

    def __init__(self, model, classes, temperature: float = 1.0):
        self.model = model
        self.classes = np.array(classes)
        self.temperature = temperature

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> TriageClassifier:
        path = Path(model_path) if model_path else paths.resolve_triage()
        if not path.exists():
            raise FileNotFoundError(
                f"No triage model at {path}.\n"
                f"  → quickest fix (offline, ~2s):  cardiag train --fixtures\n"
                f"  → or point --model / CARDIAG_DATA at an existing "
                f"triage_model.joblib."
            )
        try:
            art = joblib.load(path)
            return cls(art["model"], art["classes"], art.get("temperature", 1.0))
        except Exception as e:
            raise ValueError(
                f"{path} is not a valid cardiag triage model (expected a joblib "
                f"dict with 'model' and 'classes'). Train one with "
                f"`cardiag train --fixtures`. [{type(e).__name__}]"
            ) from None

    def triage(self, path) -> TriageResult:
        real = [str(c).lower() for c in self.classes
                if str(c).lower() not in {"unknown", "none", "nan", ""}]
        if len(real) < 2:                       # degenerate model -> don't pretend
            return TriageResult(
                file=str(path), triage="unknown",
                label="model could not be trained (need engine AND running-gear "
                      "examples in the corpus)",
                confidence=0.0, band=Band.ABSTAIN,
                band_gloss="triage model is degenerate — not a real call",
                probabilities={}, next_step="")
        # one vector per isolated span (same embedding as training), pooled in
        # probability space: no train/serve skew. See cardiag.audio.embed.
        X = model_vectors(path).vectors
        T = self.temperature
        if T and T != 1.0 and hasattr(self.model, "decision_function"):
            d = np.asarray(self.model.decision_function(X))   # temperature-calibrated
            if d.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-d / T))
                P = np.column_stack([1.0 - p1, p1])
            else:
                e = np.exp((d - d.max(1, keepdims=True)) / T)
                P = e / e.sum(1, keepdims=True)
        else:
            P = np.asarray(self.model.predict_proba(X))
        p = P.mean(0)
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
