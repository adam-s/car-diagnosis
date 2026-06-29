"""CLAP wrapper: the one expensive tier.

CLAP (``laion/clap-htsat-unfused``) gives a 512-d audio embedding and zero-shot
text-vs-audio scoring. Loaded once and reused; runs on Apple MPS when available,
else CPU. The cheap cascade (:mod:`cardiag.audio.cascade`) keeps the few percent
of audio that ever reaches this tier.
"""
from __future__ import annotations

import numpy as np
import torch

from cardiag import config

_MODEL = None
_PROC = None


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    """Lazily load CLAP once per process."""
    global _MODEL, _PROC
    if _MODEL is None:
        import sys
        from pathlib import Path
        cached = list((Path.home() / ".cache" / "huggingface" / "hub").glob(
            "*clap-htsat-unfused*"))
        if not cached:          # first run: tell the user about the ~2GB download
            print("  [cardiag] downloading the CLAP audio model (~2GB, one time; "
                  "cached afterwards)…", file=sys.stderr, flush=True)
        from transformers import ClapModel, ClapProcessor
        _MODEL = ClapModel.from_pretrained(config.CLAP_MODEL).to(_device()).eval()
        _PROC = ClapProcessor.from_pretrained(config.CLAP_MODEL)
    return _MODEL, _PROC


class Clap:
    """Thin CLAP handle. Construct once; call :meth:`score` / :meth:`embed`."""

    def __init__(self) -> None:
        self.dev = _device()
        self.m, self.p = _load()

    def score(self, clips, prompts, sr: int = config.SR_CLAP, batch: int = 24):
        """Zero-shot softmax of each clip over ``prompts`` -> (n_clips, n_prompts)."""
        out = []
        for i in range(0, len(clips), batch):
            inp = self.p(text=prompts, audio=clips[i:i + batch], sampling_rate=sr,
                         return_tensors="pt", padding=True).to(self.dev)
            with torch.no_grad():
                out.append(self.m(**inp).logits_per_audio.softmax(-1).cpu().numpy())
        return np.concatenate(out, 0)

    def embed(self, clips, sr: int = config.SR_CLAP):
        """L2-normalized 512-d embedding per clip -> (n_clips, 512)."""
        embs = []
        for y in clips:
            inp = self.p(audio=y, sampling_rate=sr, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                out = self.m.get_audio_features(**inp)
            e = out if torch.is_tensor(out) else getattr(out, "audio_embeds",
                                                          out.pooler_output)
            v = e[0].cpu().numpy()
            embs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(embs)
