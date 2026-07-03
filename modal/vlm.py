"""Modal app — an OpenAI-compatible VLM endpoint (the captioner) via vLLM.

Serves `/v1/chat/completions` accepting `image_url` content (the OpenAI Vision API),
so Plumbline records real SENSOR_TO_CAPTION calls and Experiment C can rank real
captioners by downstream decision fidelity.

    modal deploy modal/vlm.py
    # -> https://<workspace>--plumbline-vlm-serve.modal.run

Use that URL as the captioner base URL (+ `/v1`) in examples/modal_validate.py. A
template — pick a VLM that fits your GPU (see modal/README.md).
"""

import subprocess

import modal

# Qwen2-VL-2B sees objects but cannot bind them to a blocked/clear judgment
# (it captions an obstacle scene as "path is clear"); 2.5-VL-7B handles the
# perceive->judge step and still fits an A10G in fp16.
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PORT = 8000
MINUTES = 60

# Unpinned so vLLM's own consistent dependency set is used.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]", "qwen-vl-utils")
    # flashinfer's sampler JIT-compiles at first use and needs nvcc, which
    # debian_slim lacks; fall back to vLLM's built-in sampling kernels.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
app = modal.App("plumbline-vlm")


@app.function(
    image=image,
    gpu="A10G",
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=PORT, startup_timeout=15 * MINUTES)
def serve() -> None:
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--served-model-name",
            "captioner",
            "--max-model-len",
            "8192",
            "--trust-remote-code",
        ]
    )
