# FAQ / troubleshooting

Short answers to the things that trip people up on first contact.

## Which extra do I need?

The **core substrate is dependency-free** — a bare `pip install -e .` (stdlib only) is enough to load traces, run the matchers/replayer, and `plumbline gate` a committed golden episode. Add an extra only for what you actually do:

| I want to… | Install | Pulls in |
|------------|---------|----------|
| Gate a golden episode, load/replay traces (offline) | `pip install -e .` | nothing |
| Run `plumbline record` / `plumbline replay` (a proxy **server**) | `pip install -e '.[proxy]'` | httpx, uvicorn, websockets |
| Run the demos (`examples/experiment_a.py`, `experiment_c.py`) | `pip install -e '.[examples]'` | pillow, httpx |
| Use the real Zenoh bus tap (OM1 action seam) | `pip install -e '.[zenoh]'` | eclipse-zenoh |
| Use a real semantic embedder for the free-text matcher/gate | `pip install -e '.[embeddings]'` | sentence-transformers |
| Develop (lint, type-check, test) | `pip install -e '.[proxy,dev]'` | ruff, mypy, pytest, pre-commit, httpx |

Extras combine: `pip install -e '.[proxy,examples]'`. Note `dev` alone is tooling-only — it does **not** pull in the proxy server, so install `'.[proxy,dev]'` to both develop and run `record`/`replay`.

> Plumbline is not on PyPI yet, so `pip install plumbline` won't find it — install from a clone with `-e .` as above.

## `plumbline record` / `replay` raises `ModuleNotFoundError`

Those two subcommands run an ASGI proxy server, which needs the `proxy` extra:

```bash
pip install -e '.[proxy]'
```

`gate`, `diff`, `export`, and `scenes` work on a bare install. If you only meant to gate or diff a trace, you don't need the proxy at all.

## `ConnectionRefused` when I run an example

The examples call a live model endpoint — they do not ship a model. By default `examples/experiment_c.py` talks to a local **Ollama** at `http://localhost:11434/v1` with models `moondream` (VLM) and `llama3.2:1b` (decider). A refused connection means nothing is listening there. Either:

- **Start Ollama** and pull the models: `ollama serve`, then `ollama pull moondream && ollama pull llama3.2:1b`; or
- **Point at your own endpoint** via env vars — the examples read `PLUMBLINE_OLLAMA_URL` (or the split `PLUMBLINE_VLM_URL` / `PLUMBLINE_LLM_URL`) plus `PLUMBLINE_VLM` / `PLUMBLINE_DECIDER` for the model names, e.g.:

  ```bash
  PLUMBLINE_VLM_URL=https://my-host/v1  PLUMBLINE_VLM=my-captioner \
  PLUMBLINE_LLM_URL=https://my-host/v1  PLUMBLINE_DECIDER=my-cortex \
  python examples/experiment_c.py
  ```

Any OpenAI-compatible endpoint works. See each example's module docstring for its exact variables.

## `plumbline` command not found

`plumbline` is a console script installed by the package. After `pip install -e .` it lands on your PATH; run `plumbline --help`. If it's missing, your install didn't complete or the venv isn't active.

## `plumbline` command vs `python -m plumbline`

Both work. `plumbline` is the console script installed on your PATH; `python -m plumbline`
runs the same CLI via the package's `__main__` (handy when the script isn't on PATH):

```bash
plumbline gate bench/om1_gazebo_gate.py     # or:  python -m plumbline gate …
plumbline list                              # episodes on disk
```

## The gate says PASS but I changed a model — shouldn't it fail?

The gate only sees drift you express as **seam overrides** in the gate config's `build() -> GateSpec`. Point the candidate config's `overrides` at the swapped model/prompt so the counterfactual replay actually exercises it. An unchanged config gating green is correct — that's the baseline. See [api.md](api.md#plumblineregression--the-gate) and `bench/om1_gazebo_gate.py` (which also has a `build_regressed()` that gates **red**).

## Are recorded traces sensitive? Can I commit them?

Yes, sensitive. A trace stores model requests and responses **verbatim** — system
prompts, governance rules, tool outputs, and any PII they carry (the proxy does *not*
record HTTP auth headers, only bodies). Treat a trace store like source secrets:
review before committing a golden episode, and never push one to a public repo
unscrubbed. To scrub at record time, pass a redactor to the recording proxy:

```python
from plumbline.proxy import RecordingProxy, redactor_for
proxy = RecordingProxy(..., redactor=redactor_for({"api_key", "authorization"}))
```

It blanks the named JSON fields (at any depth) to `"[REDACTED]"` before anything is
written; the response returned to the runtime is unaffected (the zero-touch invariant).
Loading a trace also validates path components (episode/blob/config ids), so a
hostile shared trace can't traverse outside the store.

## More

- Concepts and the lifecycle: [concepts.md](concepts.md)
- Exact signatures: [api.md](api.md)
- What is and isn't actually built: [limitations.md](limitations.md)
