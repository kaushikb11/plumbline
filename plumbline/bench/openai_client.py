"""Real captioner / decider over an OpenAI-compatible endpoint (§4, §7 — no robot).

Lets Experiment C run with real models and no robot or simulator: point `base_url`
at a local Ollama (`http://localhost:11434/v1`, `llava` for vision + a text model
for decisions — free, no keys) or at a hosted provider. Requires httpx:
`pip install "plumbline[proxy]"`.

To make the decider's output a well-defined distribution (§7.1; the §14.6 binning
concern), the decider is prompted to answer with exactly one action from a fixed
vocabulary and the reply is normalized to that vocabulary — so `D(x)` is over a
small set of action classes, not free text.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from plumbline.bench.leaderboard import Captioner, LabeledScene
from plumbline.core.trace import JSONValue
from plumbline.fidelity import DeciderFn

DEFAULT_ACTIONS: tuple[str, ...] = ("move_forward", "turn_left", "turn_right", "back_up", "stop")
_CAPTION_INSTRUCTION = "Describe the scene in one sentence for a robot deciding how to move."


def chat_captioner(
    client: httpx.Client,
    base_url: str,
    model: str,
    *,
    instruction: str = _CAPTION_INSTRUCTION,
) -> Captioner:
    """A VLM captioner: sends the scene image to `{base_url}/chat/completions`."""
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def caption(scene: LabeledScene) -> str:
        request = {
            "model": model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": scene.image}},
                    ],
                }
            ],
        }
        response = client.post(endpoint, json=request)
        response.raise_for_status()
        return _message_content(response.json())

    return caption


def chat_decider(
    client: httpx.Client,
    base_url: str,
    model: str,
    *,
    temperature: float = 0.7,
    actions: Sequence[str] = DEFAULT_ACTIONS,
) -> DeciderFn:
    """A Cortex decider: given a context, returns an action plan from `actions`."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    system = (
        "You are a mobile robot's decision module. Given the observation, reply with "
        "exactly one action from this list and nothing else: " + ", ".join(actions) + "."
    )

    def decide(context: str) -> Mapping[str, JSONValue]:
        request = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": context},
            ],
        }
        response = client.post(endpoint, json=request)
        response.raise_for_status()
        return {"action": _normalize_action(_message_content(response.json()), actions)}

    return decide


class MalformedResponse(RuntimeError):
    """The endpoint returned 200 without a usable chat completion — e.g. an
    OpenAI-compatible `{"error": ...}` body (Ollama does this) or no `choices`."""


def _message_content(response_json: Any) -> str:
    if isinstance(response_json, dict) and "error" in response_json:
        raise MalformedResponse(f"model returned an error: {response_json['error']!r}")
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MalformedResponse(f"unexpected response shape: {response_json!r}") from exc
    if content is None:  # e.g. a tool-call-only reply has null content
        raise MalformedResponse("response has no text content (tool-call-only?)")
    return str(content)


def _normalize_action(text: str, actions: Sequence[str]) -> str:
    """Map free text to the earliest-mentioned action token (the decider is asked
    for one action; earliest-mentioned wins so 'move_forward, not stop' -> move_forward)."""
    lowered = text.strip().lower()
    best_pos = len(lowered) + 1
    best = "unknown"
    for action in actions:
        for form in (action, action.replace("_", " ")):
            pos = lowered.find(form)
            if pos != -1 and pos < best_pos:
                best_pos, best = pos, action
    return best
