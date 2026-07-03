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

MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # small vision-language model, cheap
PORT = 8000
MINUTES = 60

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.6", "huggingface_hub[hf_transfer]==0.26.2", "qwen-vl-utils==0.0.8"
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
        ]
    )
