"""Real-model captioner/decider over an OpenAI-compatible endpoint (spec §4, §7).

Drives `chat_captioner` / `chat_decider` against an httpx `MockTransport` shaped
like Ollama's OpenAI-compatible `/v1/chat/completions` — so the request framing
(vision image_url, the action prompt) and response parsing are exercised without a
real model. Point the same code at `http://localhost:11434/v1` to use real Ollama.
"""

import json

import httpx
from plumbline.bench.leaderboard import LabeledScene
from plumbline.bench.openai_client import chat_captioner, chat_decider

_SCENE = LabeledScene("s1", "data:image/png;base64,AAAA", "obstacle ahead")


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    messages = body["messages"]
    is_vision = any(isinstance(message.get("content"), list) for message in messages)
    content = "an obstacle is directly ahead" if is_vision else "I would STOP immediately."
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_handler), base_url="http://ollama.test")


def test_chat_captioner_sends_the_image_and_returns_the_caption() -> None:
    caption = chat_captioner(_client(), "http://ollama.test/v1", "llava")
    assert caption(_SCENE) == "an obstacle is directly ahead"


def test_chat_decider_normalizes_the_reply_to_the_action_vocabulary() -> None:
    decide = chat_decider(_client(), "http://ollama.test/v1", "llama3.2")
    # The model replied "I would STOP immediately." -> normalized to the "stop" class.
    assert decide("obstacle ahead") == {"action": "stop"}


def test_decider_reply_outside_the_vocabulary_is_unknown() -> None:
    def odd_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hmm, unsure"}}]})

    client = httpx.Client(transport=httpx.MockTransport(odd_handler), base_url="http://ollama.test")
    decide = chat_decider(client, "http://ollama.test/v1", "llama3.2")
    assert decide("anything") == {"action": "unknown"}
