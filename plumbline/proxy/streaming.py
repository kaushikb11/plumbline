"""Streaming (SSE) capture and replay (engineering spec §4.2, §14.3).

Token-stream responses are captured whole: the proxy stores the assembled body
plus the original chunk framing, and replay re-emits the stream with the same
chunk boundaries (some runtimes behave differently on streaming granularity).

`split_sse` preserves delimiters so that `"".join(chunks)` reproduces the raw
byte stream exactly. `assemble_openai` reassembles the delta chunks into a
non-streamed body for inspection/matching.
"""

import json
from dataclasses import dataclass

from plumbline.core.trace import JSONValue, Payload

_SSE_MARKER = "__sse__"
# SSE event delimiter: LF-LF or the CRLF-CRLF some servers emit. Each block keeps
# its actual delimiter, so `"".join(chunks)` still reproduces the raw stream.
_DELIMITERS = ("\r\n\r\n", "\n\n")


@dataclass(frozen=True)
class CapturedStream:
    """A captured SSE response: the ordered raw event blocks, framing preserved."""

    chunks: tuple[str, ...]

    @property
    def raw(self) -> str:
        return "".join(self.chunks)


def split_sse(raw: str) -> tuple[str, ...]:
    """Split a raw SSE stream into event blocks, each retaining its delimiter so
    that joining the blocks reproduces `raw` byte-for-byte."""
    blocks: list[str] = []
    rest = raw
    while rest:
        idx, delimiter = _first_delimiter(rest)
        if idx == -1:
            blocks.append(rest)
            break
        end = idx + len(delimiter)
        blocks.append(rest[:end])
        rest = rest[end:]
    return tuple(blocks)


def _first_delimiter(text: str) -> tuple[int, str]:
    """The earliest SSE delimiter in `text` and which one it is (-1 if none)."""
    best_idx = -1
    best_delimiter = ""
    for delimiter in _DELIMITERS:
        idx = text.find(delimiter)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_delimiter = idx, delimiter
    return best_idx, best_delimiter


def data_payloads(chunk: str) -> list[str]:
    """The `data:` payloads carried in one SSE event block."""
    out: list[str] = []
    for line in chunk.splitlines():
        if line.startswith("data:"):
            out.append(line[len("data:") :].strip())
    return out


def assemble_openai(stream: CapturedStream) -> JSONValue:
    """Reassemble an OpenAI-style delta stream into a single non-streamed body."""
    content: list[str] = []
    response_id: JSONValue = None
    model: JSONValue = None
    finish_reason: JSONValue = None
    for chunk in stream.chunks:
        for payload in data_payloads(chunk):
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                # A non-JSON data: line (heartbeat, non-OpenAI framing) degrades the
                # assembled inspection body, but must NOT crash record() — the raw
                # chunks (what replay re-emits) are already captured. Matches the
                # decode-guard hardening in _decode_json and the Zenoh tap.
                continue
            if isinstance(obj, dict):
                response_id = obj.get("id", response_id)
                model = obj.get("model", model)
                choices = obj.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                            content.append(str(delta["content"]))
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
    return {
        "id": response_id,
        "model": model,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content)},
                "finish_reason": finish_reason,
            }
        ],
    }


def stream_to_payload(stream: CapturedStream, assembled: JSONValue) -> Payload:
    """Wrap a captured stream as a canonical Payload: assembled body + framing."""
    return Payload(
        inline={_SSE_MARKER: True, "chunks": list(stream.chunks), "assembled": assembled}
    )


def payload_to_stream(payload: Payload) -> CapturedStream | None:
    """Recover the captured stream from a Payload, or None if not streamed."""
    inline = payload.inline
    if isinstance(inline, dict) and inline.get(_SSE_MARKER) is True:
        chunks = inline.get("chunks")
        if isinstance(chunks, list):
            return CapturedStream(tuple(str(chunk) for chunk in chunks))
    return None
