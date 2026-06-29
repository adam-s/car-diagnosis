"""Live vLLM OpenAI-compatible endpoint for Qwen2.5-14B on Modal.
Reuses the huggingface-cache Volume (weights already cached by modal_qwen.py).

  uv run --with modal modal deploy modal/serve_qwen.py
  -> prints a URL like https://<you>--mech-qwen-serve-serve.modal.run
     the OpenAI base is that URL + /v1
"""
import subprocess

import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # clear the CUDA base image's entrypoint (Modal's vLLM serving pattern)
    .uv_pip_install("vllm==0.11.0", "transformers==4.57.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("mech-qwen-serve")


@app.function(
    image=vllm_image,
    gpu="L40S",                 # 48GB fits 14B natively; ~$1.95/hr while running
    timeout=20 * 60,
    # Scale to zero when idle (no continuous GPU bill), but keep a container warm
    # for 5 minutes after the last request so back-to-back use never re-pays the
    # cold start. The first request after a fully-cold period still waits on the
    # vLLM warmup gated by startup_timeout below.
    min_containers=0,
    scaledown_window=5 * 60,    # stay warm 5 min after the last request, then release the GPU
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache},
)

@modal.concurrent(max_inputs=8)
# web_server only routes traffic once the port is live, so vLLM fully loads
# (the warmup) before any request is served, up to startup_timeout.
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve():
    # tool-calling MUST be on: Pydantic AI's structured output rides on tool calls
    subprocess.Popen(
        f"vllm serve {MODEL} --host 0.0.0.0 --port 8000 --max-model-len 8192 "
        "--enable-auto-tool-choice --tool-call-parser hermes",
        shell=True,
    )