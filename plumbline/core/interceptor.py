"""The interception interface (engineering spec §3.3).

FROZEN (CLAUDE.md invariant 1). Everything that captures events — the HTTP proxy
(§4.2), the Zenoh tap (§4.3), and any future mechanism — implements this one
Protocol, so they are interchangeable.

`maybe_replay` returning non-None means "do not call the real model, use this":
it is the hinge between record and replay (§3.3). Returning None means the call
proceeds live (and, in record mode, is captured via on_request/on_response).

NOTE: §3.3 references a `Context` type in every method signature but never
defines it. The frozen-dataclass below is the minimal faithful interpretation —
the per-call context an interceptor needs to associate a captured interaction
with its episode and model call.

`logical_tick` carries the runtime's loop-iteration index. It lives here, not on
the recorder, because a zero-touch HTTP proxy cannot infer loop boundaries from a
model call alone — only the loop driver (the runtime / adapter, or a clock hook)
knows which iteration a call belongs to, so it stamps the tick on the context.
All seams of one loop iteration share a tick, which is what lets counterfactual
replay group a swapped seam with that iteration's downstream seams (§6). The
recorder still assigns `seq` (monotonic call order) and the request digest (§3.5).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload


@dataclass(frozen=True)
class Context:
    """Per-interception context handed to an Interceptor (§3.3).

    NOTE: shape interpreted (§3.3 leaves `Context` undefined). Carries what the
    interceptor cannot derive from the (seam, payload) pair alone.
    """

    episode_id: str
    model_id: str | None
    params: Mapping[str, JSONValue]  # temperature, top_p, max_tokens, seed if any
    logical_tick: int = 0  # runtime loop-iteration index; stamped by the loop driver


class Interceptor(Protocol):
    def on_request(self, seam: Seam, request: Payload, ctx: Context) -> None: ...

    def on_response(self, seam: Seam, response: Payload, ctx: Context) -> None: ...

    # In replay mode, an interceptor may instead SERVE a response from the trace.
    def maybe_replay(self, seam: Seam, request: Payload, ctx: Context) -> Payload | None: ...
