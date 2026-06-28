"""cardiag — diagnose a car's mechanical fault from the sound it makes.

Public API
----------
    from cardiag import Classifier, TriageClassifier, clean

    clf = Classifier.load()                 # after `cardiag train --fixtures` (no model ships)
    result = clf.diagnose("clip.wav")       # cleans -> embeds -> heads
    result.verdict                          # Verdict.FAULT | NORMAL | UNCERTAIN
    result.fault_probability                # 0.81
    result.causes                           # [Cause(part="wheel bearing", p=0.34), ...]
    result.to_dict()                        # JSON-ready

    clean("clip.wav")                       # isolate mechanical sound (no model)
    TriageClassifier.load().triage("clip.wav")   # calibrated engine-vs-running-gear

The pipeline that produces the training corpus lives under ``cardiag.ingest`` /
``cardiag.pipeline`` / ``cardiag.training``.
"""
from cardiag.audio import CleanResult, clean
from cardiag.inference import Classifier, TriageClassifier
from cardiag.types import (
    Band,
    Cause,
    Diagnosis,
    Segment,
    TriageResult,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "Classifier",
    "TriageClassifier",
    "clean",
    "CleanResult",
    "Diagnosis",
    "Cause",
    "Segment",
    "TriageResult",
    "Verdict",
    "Band",
    "__version__",
]
