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

## `python -m plumbline` doesn't work — use the console script

`plumbline` is a package without a `__main__`, so `python -m plumbline` currently errors (`No module named plumbline.__main__`). Use the installed console script instead:

```bash
plumbline gate bench/om1_gazebo_gate.py     # not:  python -m plumbline gate …
plumbline list                              # episodes on disk
```

## The gate says PASS but I changed a model — shouldn't it fail?

The gate only sees drift you express as **seam overrides** in the gate config's `build() -> GateSpec`. Point the candidate config's `overrides` at the swapped model/prompt so the counterfactual replay actually exercises it. An unchanged config gating green is correct — that's the baseline. See [api.md](api.md#plumblineregression--the-gate) and `bench/om1_gazebo_gate.py` (which also has a `build_regressed()` that gates **red**).

## More

- Concepts and the lifecycle: [concepts.md](concepts.md)
- Exact signatures: [api.md](api.md)
- What is and isn't actually built: [limitations.md](limitations.md)
