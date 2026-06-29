"""``clean()``: isolate the mechanical sound from a recording.

This is the function the project description calls for: *"the uploaded clip is
cleaned with removal of music, voice, and other sounds, exactly as in the
training process."* It runs the identical cascade
(:mod:`cardiag.audio.cascade`) used to build the corpus, plus the CLAP music
gate, so an uploaded clip and a training clip are processed the same way.

It handles the hard cases the corpus is full of:
  * a 10-minute explainer with only ~15s of actual fault sound  -> the cascade
    isolates the few loud, non-speech spans and discards the narration;
  * a TikTok that is only background music                      -> the music
    gate flags it (``is_music``) and drops the musical spans;
  * a compilation of several different noises                   -> returns one
    :class:`~cardiag.types.Segment` per distinct span.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

from cardiag import config
from cardiag.audio.cascade import candidate_regions
from cardiag.types import Segment

# CLAP music-gate prompts + threshold (from the corpus music_gate stage).
_MUSIC_PROMPTS = ["music or a song with a beat",
                  "a mechanical car noise or engine sound",
                  "a person talking", "silence or ambient noise"]
MUSIC_THRESH = 0.5


@dataclass
class CleanResult:
    """What :func:`clean` isolated from one recording."""
    file: str
    segments: list[Segment]
    speech_fraction: float
    total_seconds: float
    music_probability: float
    sr: int
    isolated: list[np.ndarray] = field(default_factory=list)

    @property
    def kept_seconds(self) -> float:
        return round(sum(s.duration for s in self.segments), 3)

    @property
    def is_empty(self) -> bool:
        """No mechanical span survived (all speech / silence / music)."""
        return not self.segments

    @property
    def is_music(self) -> bool:
        """The recording is dominated by music, not a mechanical sound."""
        return self.music_probability >= MUSIC_THRESH

    def merged_audio(self) -> np.ndarray:
        """All isolated spans concatenated: the clean signal to embed."""
        return (np.concatenate(self.isolated) if self.isolated
                else np.zeros(0, dtype=np.float32))

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "total_seconds": round(self.total_seconds, 2),
            "kept_seconds": self.kept_seconds,
            "speech_fraction": self.speech_fraction,
            "music_probability": round(self.music_probability, 3),
            "is_music": self.is_music,
            "is_empty": self.is_empty,
            "segments": [s.to_dict() for s in self.segments],
        }


def clean(path, *, music_gate: bool = True, sr: int = config.SR_CLAP,
          max_segments: int | None = None) -> CleanResult:
    """Isolate mechanical spans from ``path``.

    Parameters
    ----------
    path : str | Path
        Audio file (any format librosa/ffmpeg can read).
    music_gate : bool
        Score each surviving span with CLAP and drop musical ones (default on).
    sr : int
        Sample rate of the returned isolated audio (CLAP's 48 kHz by default).
    max_segments : int | None
        Keep at most this many of the longest spans.
    """
    import warnings
    if not Path(path).exists():
        raise FileNotFoundError(f"no such audio file: {path}")
    try:
        with warnings.catch_warnings():       # hush librosa's audioread fallback noise
            warnings.simplefilter("ignore")
            y16, sr16 = librosa.load(str(path), sr=config.SR_CHEAP, mono=True)
            yhi, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        raise ValueError(
            f"could not read audio from {path} — is it a valid audio file? "
            f"({type(exc).__name__})") from None
    # defensively sanitize NaN/inf (a corrupt clip should not poison the cascade)
    y16 = np.nan_to_num(y16, copy=False)
    yhi = np.nan_to_num(yhi, copy=False)
    if y16.size == 0:                       # empty/garbled file -> nothing to isolate
        return CleanResult(file=str(path), segments=[], speech_fraction=0.0,
                           total_seconds=0.0, music_probability=0.0, sr=sr,
                           isolated=[])
    total_seconds = len(y16) / sr16
    regions, speech_fraction = candidate_regions(y16, int(sr16), return_speech_frac=True)

    segs: list[Segment] = []
    isolated: list[np.ndarray] = []
    for (s, e) in regions:
        seg_audio = yhi[int(s * sr):int(e * sr)]
        if len(seg_audio) < sr // 2:
            continue
        flat = float(np.mean(librosa.feature.spectral_flatness(y=seg_audio)))
        segs.append(Segment(start=s, end=e, speech_coverage=0.0, flatness=round(flat, 4)))
        isolated.append(seg_audio)

    music_probability = 0.0
    if music_gate and isolated:
        from cardiag.audio.clap import Clap
        scores = Clap().score(isolated, _MUSIC_PROMPTS, sr=sr)
        music_probability = float(scores[:, 0].max())
        # drop the musical spans; keep the mechanical ones
        keep = [i for i, row in enumerate(scores) if float(row[0]) < MUSIC_THRESH]
        segs = [segs[i] for i in keep]
        isolated = [isolated[i] for i in keep]

    if max_segments is not None and len(segs) > max_segments:
        order = sorted(range(len(segs)), key=lambda i: -segs[i].duration)[:max_segments]
        order.sort()
        segs = [segs[i] for i in order]
        isolated = [isolated[i] for i in order]

    return CleanResult(
        file=str(path),
        segments=segs,
        speech_fraction=speech_fraction,
        total_seconds=total_seconds,
        music_probability=music_probability,
        sr=sr,
        isolated=isolated,
    )
