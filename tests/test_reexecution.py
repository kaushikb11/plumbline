"""True re-execution determinism (eng spec §15, §3.6, §4.2).

`test_determinism.py` verifies serialization fidelity (record -> store -> reload
is byte-identical). This test verifies the stronger property the project actually
claims: re-driving a runtime loop while the proxy serves each recorded model
response *by request_digest* reproduces the same decision/action sequence — even
though the models are nondeterministic.

The loop makes two model calls per tick (caption, decide) and recomputes the
deterministic transforms (fuse, act) in between. In record mode the calls hit the
stub models through the RecordingProxy; in replay mode the SAME loop runs but each
call is served from the trace by the ReplayingProxy. If serving works, the
replayed action sequence equals the recorded one; a fresh live run (different
seed) must differ, proving the serving is doing the work.
"""

import random
from collections.abc import Callable

from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload
from plumbline.proxy import RecordingProxy, ReplayingProxy

from tests.toyloop import DEFAULT_RULES, StubCaptioner, StubDecider, fuse, make_frames

ModelCall = Callable[[Payload], Payload]
_CTX = Context(episode_id="reexec", model_id="stub/model", params={"temperature": 0.8})


def _caption_of(response: Payload) -> str:
    inline = response.inline
    assert isinstance(inline, dict)
    caption = inline["caption"]
    assert isinstance(caption, str)
    return caption


def _plan_of(response: Payload) -> JSONValue:
    inline = response.inline
    assert isinstance(inline, dict)
    return inline["action_plan"]


def _seam_classifier(request: Payload, ctx: Context) -> Seam:
    inline = request.inline
    if isinstance(inline, dict) and inline.get("kind") == "caption":
        return Seam.SENSOR_TO_CAPTION
    return Seam.FUSE_TO_DECIDE


def _make_upstream(captioner: StubCaptioner, decider: StubDecider) -> ModelCall:
    """The live model endpoint: dispatch caption/decide requests to the stubs."""
    frames = make_frames()

    def upstream(request: Payload) -> Payload:
        inline = request.inline
        assert isinstance(inline, dict)
        if inline["kind"] == "caption":
            tick = inline["tick"]
            assert isinstance(tick, int)
            return Payload(inline={"caption": captioner.caption(frames[tick])})
        prompt = inline["prompt"]
        assert isinstance(prompt, str)
        return Payload(inline={"action_plan": decider.decide(prompt)})

    return upstream


def _drive(call_model: ModelCall) -> tuple[JSONValue, ...]:
    """Run the loop, obtaining each model response via `call_model`. The fuse and
    act transforms are deterministic, so the only nondeterminism is in the served
    model responses."""
    actions: list[JSONValue] = []
    for tick in range(len(make_frames())):
        caption = _caption_of(call_model(Payload(inline={"kind": "caption", "tick": tick})))
        prompt = fuse([caption], DEFAULT_RULES)
        plan = _plan_of(call_model(Payload(inline={"kind": "decide", "prompt": prompt})))
        actions.append(plan)
    return tuple(actions)


def test_replay_reexecution_reproduces_actions_while_live_run_diverges() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())

    # Record: drive the loop with the proxy capturing every served model call.
    record_upstream = _make_upstream(
        StubCaptioner(random.Random(7), 0.8), StubDecider(random.Random(7), 0.8)
    )
    proxy = RecordingProxy(record_upstream, recorder, classifier=_seam_classifier)
    recorded_actions = _drive(lambda req: proxy.forward(req, _CTX))
    proxy.close(_CTX.episode_id)

    # Replay: re-drive the SAME loop, serving each model call from the trace.
    replay = ReplayingProxy(store, _CTX.episode_id)
    replayed_actions = _drive(lambda req: replay.faithful(req, _CTX))

    assert replayed_actions == recorded_actions
    assert len(recorded_actions) == len(make_frames())

    # A fresh live run with a different seed diverges — so the equality above is a
    # real reproduction, not a trivial determinism of the loop.
    live_upstream = _make_upstream(
        StubCaptioner(random.Random(99), 0.8), StubDecider(random.Random(99), 0.8)
    )
    live_actions = _drive(live_upstream)
    assert live_actions != recorded_actions
