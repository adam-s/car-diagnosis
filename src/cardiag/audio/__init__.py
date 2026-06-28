"""Audio layer: the CLAP wrapper, the cheap cleaning cascade, and the public
``clean()`` used identically at corpus-build time and at inference."""
from cardiag.audio.cascade import (
    candidate_regions,
    cyclic_features,
    spectral_fingerprint,
)
from cardiag.audio.clean import CleanResult, clean

__all__ = [
    "clean",
    "CleanResult",
    "candidate_regions",
    "cyclic_features",
    "spectral_fingerprint",
]
