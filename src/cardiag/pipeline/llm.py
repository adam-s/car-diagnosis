"""One swappable LLM interface for the whole pipeline — Modal / Ollama / Haiku.

The cost lesson: per-call Haiku adds up at scale. This routes batch LLM work to
the cheapest adequate backend:
  ollama  — local Qwen on this Mac, $0 (default; contends with CLAP, slower)
  modal   — Qwen2.5 on a Modal GPU via vLLM, cents/run (fast, set-and-forget)
  claude  — Haiku via `claude -p` (fallback / highest quality)

    from cardiag.pipeline.llm import run_batch
    results = run_batch([("id1","prompt1"), ...], backend="modal")
    # -> {"id1": "completion text", ...}
"""
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cardiag import paths

OLLAMA_MODEL = "qwen2.5:7b-instruct"
CLAUDE_MODEL = "claude-haiku-4-5"


def _ollama_one(prompt):
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt,
                       "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["response"]


def _claude_one(prompt):
    return subprocess.run(["claude", "-p", "--model", CLAUDE_MODEL, prompt],
                          capture_output=True, text=True, timeout=180).stdout


def _parallel(items, fn, workers):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, p): i for i, p in items}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = ""
    return out


def run_batch(items, backend="ollama", workers=6):
    """items: list of (id, prompt). Returns {id: completion}."""
    if backend == "ollama":
        return _parallel(items, _ollama_one, workers)
    if backend == "claude":
        return _parallel(items, _claude_one, workers)
    if backend == "modal":
        # write prompts, run the Modal batch app, read completions
        d = paths.TRAIN_DATA
        pin, pout = d / "llm_prompts.jsonl", d / "llm_out.jsonl"
        with open(pin, "w") as fh:
            for i, p in items:
                fh.write(json.dumps({"id": i, "prompt": p}) + "\n")
        modal_qwen = Path(__file__).resolve().parent.parent / "modal" / "modal_qwen.py"
        subprocess.run(
            ["uv", "run", "--with", "modal", "modal", "run",
             str(modal_qwen),
             "--input", str(pin), "--output", str(pout)],
            check=True)
        return {json.loads(l)["id"]: json.loads(l)["text"]
                for l in open(pout)}
    raise ValueError(f"unknown backend {backend}")


def parse_json(text):
    """Best-effort extract the first JSON object/array from a completion."""
    for a, b in (("{", "}"), ("[", "]")):
        i, j = text.find(a), text.rfind(b)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                pass
    return None
