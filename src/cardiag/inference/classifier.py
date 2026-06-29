"""``Classifier``: turn one recording into a structured :class:`Diagnosis`.

Loads the clean-teacher heads (trained by ``training/models/train_best.py``)
over frozen CLAP embeddings: ``kind`` (fault vs normal), ``knock`` (engine knock
vs normal), and ``cause`` (part family). Honest by design: cause is top-k with
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
from cardiag.audio.embed import model_vectors
from cardiag.types import Cause, Diagnosis, Region, Verdict

CAUSE_HELP = {
    "brakes": "worn brake pads/rotors — inspect brakes",
    "belt": "serpentine/accessory belt or tensioner",
    "power_steering": "power-steering pump or fluid",
    "accessories": "alternator / starter / battery / A-C",
    "fuel_ignition": "ignition or fuel delivery (hard start)",
    "low_oil": "low oil / top-end — check oil level NOW",
}

_FAULT_HI, _FAULT_LO = 0.6, 0.4


def _proba(clf, X, temperature: float = 1.0) -> dict:
    """Mean class probability over every span vector of one recording.

    Pooling happens here, in probability space: each row of ``X`` is a single
    in-distribution span embedding, scored independently, then averaged. We never
    average the embeddings themselves (that would be a vector the head never saw
    at fit time, the train/serve skew this design avoids).

    ``temperature`` (fit at train time, Guo et al. 2017) divides the head's logits
    before the softmax/sigmoid so a weak head stops being over-confident: the
    decision is unchanged, only the reported probability. Falls back to plain
    ``predict_proba`` for heads without logits (e.g. a degenerate DummyClassifier)
    or T==1.
    """
    if temperature and temperature != 1.0 and hasattr(clf, "decision_function"):
        d = np.asarray(clf.decision_function(X))
        if d.ndim == 1:                       # binary: sigmoid(logit / T)
            p1 = 1.0 / (1.0 + np.exp(-d / temperature))
            P = np.column_stack([1.0 - p1, p1])   # cols align with classes_ = [neg, pos]
        else:                                 # multiclass: softmax(scores / T)
            e = np.exp((d - d.max(1, keepdims=True)) / temperature)
            P = e / e.sum(1, keepdims=True)
    else:
        P = np.asarray(clf.predict_proba(X))
    return dict(zip(clf.classes_, P.mean(0)))


_SENTINELS = {"unknown", "none", "nan", ""}


def _usable(head) -> bool:
    """A head carries information only if it has >=2 real (non-placeholder) classes."""
    classes = [str(c).lower() for c in getattr(head, "classes_", [])]
    real = [c for c in classes if c not in _SENTINELS]
    return len(real) >= 2


class Classifier:
    """Bundle of the three trained heads. Construct via :meth:`load`."""

    def __init__(self, heads: dict, temps: dict | None = None):
        self.heads = heads
        self.temps = temps or {}

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> Classifier:
        """Load heads from a joblib artifact (defaults to the bundled model)."""
        path = Path(model_path) if model_path else paths.resolve_clap()
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}.\n"
                f"  → quickest fix (offline, ~2s):  cardiag train --fixtures\n"
                f"  → use the shipped model:  cardiag serve --model models  (or copy "
                f"models/*.joblib into data/training/)\n"
                f"  → real model:  cardiag scrape youtube && cardiag train"
            )
        try:
            art = joblib.load(path)
            heads = art["heads"]
            assert {"kind", "knock", "cause"} <= set(heads)
        except Exception as e:
            raise ValueError(
                f"{path} is not a valid cardiag diagnosis model "
                f"(expected a joblib dict with a 'heads' map of kind/knock/cause). "
                f"Train one with `cardiag train --fixtures`. [{type(e).__name__}]"
            ) from None
        return cls(heads, art.get("temps", {}))

    def diagnose(self, path, *, clean_audio: bool = True) -> Diagnosis:
        """Diagnose ``path``: clean -> embed -> heads -> :class:`Diagnosis`.

        The clip becomes one vector per isolated span (each embedded exactly as a
        training clip was, :func:`cardiag.audio.embed.model_vectors`) and every
        head's probabilities are pooled across those spans. Train and serve feed
        the heads the same kind of vector, so there is no train/serve skew.

        Honest about degenerate heads: a head trained on a single class (or on a
        placeholder label) carries no information, so we never present its output
        as a confident verdict/cause: we downgrade to UNCERTAIN and say so.
        """
        try:
            ev = model_vectors(path, clean_audio=clean_audio)
        except ValueError:
            # too short or near-silent / unreadable -> degrade honestly, don't crash
            # or emit a confident verdict on silence (CLAP embeds silence near the
            # fault cluster, which would otherwise read as a confident FAULT).
            return Diagnosis(
                file=str(path), verdict=Verdict.UNCERTAIN, fault_probability=0.0,
                engine_knock_probability=0.0, causes=[], segments=[],
                note="clip is too short or has no usable (non-silent) audio to diagnose. "
                     + Diagnosis.note)
        X, segments, res = ev.vectors, ev.segments, ev.clean_result
        notes = []

        # --- fault/normal -----------------------------------------------------
        if _usable(self.heads["kind"]):
            kp = _proba(self.heads["kind"], X, self.temps.get("kind", 1.0))
            p_fault = float(kp.get("fault", 0.0))
            verdict = (Verdict.FAULT if p_fault >= _FAULT_HI else
                       Verdict.NORMAL if p_fault <= _FAULT_LO else Verdict.UNCERTAIN)
        else:
            p_fault, verdict = 0.0, Verdict.UNCERTAIN
            notes.append("fault/normal head was trained on a single class — verdict "
                         "is not meaningful (scrape both fault and normal clips).")

        # --- engine knock probability (needed to gate the knock specialist below) -
        knock_classes = set(getattr(self.heads["knock"], "classes_", []))
        p_knock = 0.0
        if "knock" in knock_classes:
            kn = _proba(self.heads["knock"], X, self.temps.get("knock", 1.0))
            p_knock = float(kn.get("knock", 0.0))

        # --- where in the car (region): the headline localization, OOS-robust --
        # Coarse-to-fine cascade: a knock-sound is one label worn by many causes, so
        # when a knock is likely we SOFT-blend in a region head trained only on knock
        # clips (it localizes knocks better). Soft gating by p_knock, not a hard
        # switch, so a shaky knock detector can't break the general path.
        regions: list[Region] = []
        region_head = self.heads.get("region")
        if region_head is not None and _usable(region_head):
            rp = _proba(region_head, X, self.temps.get("region", 1.0))
            kr_head = self.heads.get("knock_region")
            if p_knock > 0.0 and kr_head is not None and _usable(kr_head):
                krp = _proba(kr_head, X, self.temps.get("knock_region", 1.0))
                rp = {z: (1 - p_knock) * rp.get(z, 0.0) + p_knock * krp.get(z, 0.0)
                      for z in set(rp) | set(krp)}     # gate: weight specialist by P(knock)
            regions = [Region(zone=z, p=round(float(p), 3))
                       for z, p in sorted(rp.items(), key=lambda kv: -kv[1])[:3]
                       if str(z).lower() not in _SENTINELS]

        # --- cause (finer part shortlist) -------------------------------------
        if _usable(self.heads["cause"]):
            cause = _proba(self.heads["cause"], X, self.temps.get("cause", 1.0))
            topk = sorted(cause.items(), key=lambda kv: -kv[1])[:3]
            causes = [Cause(part=k, p=round(float(v), 3), note=CAUSE_HELP.get(k, ""))
                      for k, v in topk]
        else:
            causes = []
            notes.append("cause head has too few classes to suggest a part.")

        # (engine-knock probability p_knock is computed above, where it gates the
        # knock-region specialist; it is reported but NOT the headline: the binary
        # knock head did not generalize out-of-sample, 0.99 in-dist -> 0.56 OOS.)
        if res is not None and getattr(res, "is_music", False):
            notes.append("Recording looks like mostly music — diagnosis is unreliable.")
        elif res is not None and getattr(res, "is_empty", False):
            notes.append("Cleaning isolated no clear mechanical sound; diagnosed the "
                         "whole clip.")
        note = " ".join(notes + [Diagnosis.note])

        return Diagnosis(
            file=str(path),
            verdict=verdict,
            fault_probability=p_fault,
            engine_knock_probability=p_knock,
            regions=regions,
            causes=causes,
            segments=segments,
            note=note,
        )
