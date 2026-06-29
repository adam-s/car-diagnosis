"""Audio-LLM second opinion on Modal: Qwen2-Audio-7B-Instruct via vLLM.

Why: the region classifier is confidently WRONG on the engine minority
(confidence.py band table). An audio-LLM has a different inductive bias
(AudioSet + speech pretraining, instruction-following) and LISTENS to the clip
directly: agreement between it and the CLAP head is a confidence signal
neither gives alone, and at scale it is a cross-modal label verifier
(text-mined label vs what the audio actually contains).

Same proven image pinning as modal_qwen.py (vLLM 0.11.0 + transformers 4.57,
goblins config); audio arrives as 16k PCM16 wav bytes, decoded with stdlib
`wave` (no audio libs needed in image beyond numpy).

  modal run modal/modal_audio.py --manifest data/training/smoke/manifest.jsonl \
      --output data/training/smoke/qwen_audio.jsonl
    manifest: {"id": str, "wav": path} per line (extra keys ignored)
    output  : {"id": str, "text": str} per line
"""
import json

import modal

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
MAX_TOKENS = 220

# v2: de-biased. The v1 prompt led with engine examples and put grinding/
# knocking first; the model collapsed to a constant "engine/grinding/0.8".
# This forces it to TRANSCRIBE the sound first, lists chassis first, and gives
# parallel evidence for both regions so neither anchors the answer.
PROMPT = (
    "You are an expert auto mechanic with a trained ear. Listen carefully to "
    "this recording of a car noise. Do not assume — base every field only on "
    "what you actually hear.\n"
    "Step 1: in 'heard', describe the sound in 3-6 words (pitch, rhythm, "
    "texture; e.g. 'low rhythmic clunk', 'steady high whir', 'fast metallic "
    "tick').\n"
    "Step 2: decide where it comes from.\n"
    "  CHASSIS noises (wheels/suspension/driveline/steering/brakes): a "
    "wheel-bearing hum or growl that rises with speed, a suspension or ball-"
    "joint clunk over bumps, a CV-joint click while turning, a brake squeal "
    "or grind.\n"
    "  ENGINE-INTERNAL noises (in the running motor, steady with RPM not "
    "road speed): a deep rod/main-bearing knock, a lighter lifter/valvetrain "
    "tick, a low-oil top-end rattle.\n"
    "Respond with ONLY a raw JSON object, no prose:\n"
    '{"heard": "<3-6 words>", '
    '"sound": one of [humming, whining, grinding, squealing, clicking, '
    'clunking, rattling, ticking, knocking, hissing, normal], '
    '"region": "chassis" or "engine", '
    '"part": best specific guess, '
    '"confidence": 0.0-1.0 (how sure of region)}'
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .uv_pip_install("vllm==0.11.0", "transformers==4.57.0",
                    "huggingface_hub[hf_transfer]", "librosa", "soundfile")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("mech-qwen-audio")


@app.cls(gpu="L40S", image=vllm_image, timeout=30 * 60,
         scaledown_window=60, max_containers=1,
         volumes={"/root/.cache/huggingface": hf_cache,
                  "/root/.cache/vllm": vllm_cache})
class QwenAudio:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=MODEL, max_model_len=4096,
                       gpu_memory_utilization=0.90,
                       limit_mm_per_prompt={"audio": 1})

    @modal.method()
    def generate(self, items):
        """items: [(id, wav_bytes)] of 16k mono PCM16 wavs -> [(id, text)]."""
        import io
        import wave

        import numpy as np
        from vllm import SamplingParams

        def decode(b):
            with wave.open(io.BytesIO(b)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
                return pcm.astype(np.float32) / 32768.0, w.getframerate()

        tmpl = ("<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|>"
                f"<|audio_eos|>\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n")
        reqs = [{"prompt": tmpl,
                 "multi_modal_data": {"audio": [decode(b)]}}
                for _, b in items]
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        outs = self.llm.generate(reqs, sp)
        return [(i, o.outputs[0].text) for (i, _), o in zip(items, outs)]


@app.local_entrypoint()
def main(manifest: str, output: str, batch: int = 256):
    rows = [json.loads(l) for l in open(manifest)]
    print(f"{len(rows)} clips -> L40S {MODEL}")
    qa = QwenAudio()
    with open(output, "w") as fh:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            items = [(r["id"], open(r["wav"], "rb").read()) for r in chunk]
            for cid, text in qa.generate.remote(items):
                fh.write(json.dumps({"id": cid, "text": text}) + "\n")
            fh.flush()
            print(f"  {min(i + batch, len(rows))}/{len(rows)}")
    print(f"wrote {output}")
