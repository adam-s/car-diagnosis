"""Typed result objects: the public, self-documenting API surface.

Every front-end (CLI, web, library use) receives these dataclasses, never bare
dicts. ``to_dict()`` gives a JSON-ready view for serialization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """Coarse fault call from the ``kind`` head."""
    FAULT = "fault"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"


class Band(str, Enum):
    """Calibrated confidence band, glossed from the held-out reliability table."""
    HIGH = "high"          # ~90% of similar held-out cases were correct
    MEDIUM = "medium"      # ~84%
    LOW = "low"            # a lean, not a call; verify
    ABSTAIN = "abstain"    # too close to call from this audio


@dataclass(frozen=True)
class Segment:
    """One clean mechanical span isolated from a recording by the cascade.

    ``start``/``end`` are seconds in the source clip. The cascade drops silence,
    speech, static and music before a span becomes a Segment.
    """
    start: float
    end: float
    speech_coverage: float = 0.0
    flatness: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        return {**asdict(self), "duration": self.duration}


@dataclass(frozen=True)
class Region:
    """A 'where in the car' zone with its probability. This is the headline
    localization output: the OOS sanity check shows the right zone lands in the
    top-3 ~75% of the time on held-out verified clips, where fine cause and knock
    do not generalize."""
    zone: str
    p: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Cause:
    """A candidate part/cause with its probability. Cause is *suggestive*, not
    definitive: fine cause from audio alone has a measured ceiling."""
    part: str
    p: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Diagnosis:
    """The full result of :meth:`cardiag.inference.Classifier.diagnose`."""
    file: str
    verdict: Verdict
    fault_probability: float
    engine_knock_probability: float
    regions: list[Region] = field(default_factory=list)
    causes: list[Cause] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    note: str = ("Fine cause from sound alone is uncertain; treat as triage, "
                 "not a final diagnosis.")

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "verdict": self.verdict.value,
            "fault_probability": round(self.fault_probability, 3),
            "regions": [r.to_dict() for r in self.regions],
            "causes": [c.to_dict() for c in self.causes],
            "engine_knock_probability": round(self.engine_knock_probability, 3),
            "segments": [s.to_dict() for s in self.segments],
            "note": self.note,
        }


@dataclass(frozen=True)
class TriageResult:
    """Coarse engine-internal vs running-gear call with a calibrated band: the
    one acoustically separable distinction, the honest headline product."""
    file: str
    triage: str               # "engine" | "chassis"
    label: str                # human-readable gloss
    confidence: float
    band: Band
    band_gloss: str
    probabilities: dict[str, float] = field(default_factory=dict)
    next_step: str = ""

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "triage": self.triage,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "band": self.band.value,
            "band_gloss": self.band_gloss,
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "next_step": self.next_step,
        }
