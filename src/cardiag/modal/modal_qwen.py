"""Batch LLM inference on Modal: Qwen2.5 on a GPU via vLLM.

Image/version pinning copied from the proven goblins setup
(~/Projects/goblins/.../infra/modal/qwen_grader.py): vLLM 0.11.0 + transformers
4.57.0 on a CUDA base. (vLLM 0.6.3 crashed on Qwen2.5's rope_scaling config.)
Reuses the same `huggingface-cache` Volume, so the 15GB weights are already
cached and cold start is just the vLLM load.

Batch jsonl in/out so the labeler stays backend-agnostic (pipeline/llm.py):
  modal run modal/modal_qwen.py --input <prompts.jsonl> --output <out.jsonl>
    prompts: {"id": str, "prompt": str} per line
    output : {"id": str, "text": str}  per line
"""
import json

import modal

# 14B for label quality: fits comfortably on the L40S (48GB); label quality is
# this project's binding constraint, and 14B is sharper than 7B on the extraction.
MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAX_TOKENS = 768

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .uv_pip_install("vllm==0.11.0", "transformers==4.57.0",
                    "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("mech-qwen-batch")


@app.cls(gpu="L40S", image=vllm_image, timeout=30 * 60,
         scaledown_window=60, max_containers=1,
         volumes={"/root/.cache/huggingface": hf_cache,
                  "/root/.cache/vllm": vllm_cache})
class Qwen:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=MODEL, max_model_len=8192,
                       gpu_memory_utilization=0.90)

    @modal.method()
    def generate(self, prompts):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self.llm.chat(msgs, sp)
        return [o.outputs[0].text for o in outs]


@app.local_entrypoint()
def main(input: str, output: str, batch: int = 512):
    rows = [json.loads(l) for l in open(input)]
    print(f"{len(rows)} prompts -> L40S {MODEL}")
    qwen = Qwen()
    with open(output, "w") as fh:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            texts = qwen.generate.remote([r["prompt"] for r in chunk])
            for r, t in zip(chunk, texts):
                fh.write(json.dumps({"id": r["id"], "text": t}) + "\n")
            fh.flush()
            print(f"  {min(i + batch, len(rows))}/{len(rows)}")
    print(f"wrote {output}")
