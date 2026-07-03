"""Modal app — an OpenAI-compatible LLM endpoint (the OM1 "Cortex" decider) via vLLM.

Serves `/v1/chat/completions` with tool/function calling, so Plumbline can record
real, nondeterministic (temperature > 0) FUSE_TO_DECIDE calls and prove faithful
replay reproduces them byte-for-byte against a REAL model.

    modal deploy modal/llm.py
    # -> https://<workspace>--plumbline-llm-serve.modal.run

Use that URL as the decider base URL (+ `/v1`) in examples/modal_validate.py. This is
a template: pick a model your GPU fits and a vLLM/tool-parser combo that matches it
(see modal/README.md). Cost is scale-to-zero + per-second GPU — a few dollars.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-3B-Instruct"  # small, tool-calling capable, cheap on an A10G
PORT = 8000
MINUTES = 60

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.6", "huggingface_hub[hf_transfer]==0.26.2"
)
app = modal.App("plumbline-llm")


@app.function(
    image=image,
    gpu="A10G",
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,
)
@modal.concurrent(max_inputs=32)
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
            "cortex",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",  # Qwen2.5 tool-call format; change for a different model family
        ]
    )
