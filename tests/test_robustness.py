"""Production-robustness regressions found in the framework review: the record
path must never break the runtime's forward path (zero-touch, invariant 4), a
corrupt trace line must fail loudly not silently, faithful replay must detect
UNDER-consumption, and the manifest must not be rewritten per event."""

import pytest
from plumbline.core.clock import VirtualClock
from plumbline.core.interceptor import Context
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonicalize
from plumbline.proxy.recording import ProxyDivergence, RecordingProxy, ReplayingProxy

_CTX = Context(episode_id="ep", model_id=None, params={}, logical_tick=0)


def _payload(x: object) -> Payload:
    return Payload(inline={"v": x})  # type: ignore[dict-item]


class _ExplodingRecorder(Recorder):
    def record(self, event: SeamEvent) -> None:
        raise OSError("disk full")


def test_recording_failure_does_not_break_the_forward_path() -> None:
    # Zero-touch (invariant 4): a record-layer fault must NOT turn the good upstream
    # response into an error the runtime sees.
    store = TraceStore()
    recorder = _ExplodingRecorder(store, VirtualClock())
    recorder.open_episode("ep", {})
    proxy = RecordingProxy(upstream=lambda req: _payload("UPSTREAM_OK"), recorder=recorder)
    proxy._opened.add("ep")  # pretend open (open_episode already ran above)
    proxy._seq["ep"] = 0
    served = proxy.forward(_payload("req"), _CTX)
    assert served.inline == {"v": "UPSTREAM_OK"}  # the runtime got its response


def test_corrupt_events_line_fails_loudly_with_location() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep", {})
    req = _payload("r")
    recorder.record(
        SeamEvent(
            "ep", 0, Seam.FUSE_TO_DECIDE, 0, 0.0, req, req, None, {}, canonicalize(req).digest, 0.0
        )
    )
    recorder.close_episode("ep")
    # Corrupt the trace: append a truncated JSON line.
    events_path = store.root / "episodes" / "ep" / "events.jsonl"
    events_path.write_text(events_path.read_text() + '{"seq": 1, "seam":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt event at events.jsonl line 2"):
        store.load_episode("ep")


def test_manifest_not_rewritten_per_event_but_final_on_close() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep", {"robot": "go2"})
    for seq in range(5):
        req = _payload(seq)
        recorder.record(
            SeamEvent(
                "ep",
                seq,
                Seam.FUSE_TO_DECIDE,
                seq,
                0.0,
                req,
                req,
                None,
                {},
                canonicalize(req).digest,
                0.0,
            )
        )
    # Mid-record: metadata + events readable (events.jsonl is authoritative).
    mid = store.load_episode("ep")
    assert mid.metadata == {"robot": "go2"} and len(mid.events) == 5
    recorder.close_episode("ep")
    # After close: the manifest's seam index is complete.
    assert len(store.load_manifest("ep").seam_index) == 5


def test_blob_content_addressed_stored_once() -> None:
    # The dedup the store claims (put_blob: `if not path.exists()`) — pin that
    # identical bytes land as ONE file, not two.
    from plumbline.core.trace import BlobKind

    store = TraceStore()
    store.put_blob(b"same-bytes", BlobKind.BIN)
    store.put_blob(b"same-bytes", BlobKind.BIN)
    store.put_blob(b"other-bytes", BlobKind.BIN)
    assert len(list((store.root / "blobs").iterdir())) == 2


def _record_two_calls(store: TraceStore) -> None:
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep", {})
    for seq, action in enumerate(("a", "b")):
        req = _payload(f"req-{action}")
        recorder.record(
            SeamEvent(
                "ep",
                seq,
                Seam.FUSE_TO_DECIDE,
                seq,
                0.0,
                req,
                _payload(action),
                None,
                {},
                canonicalize(req).digest,
                0.0,
            )
        )
    recorder.close_episode("ep")


def test_faithful_replay_detects_under_consumption() -> None:
    # A runtime that issues FEWER calls than recorded must NOT be reported clean.
    store = TraceStore()
    _record_two_calls(store)

    replay = ReplayingProxy(store, "ep")
    replay.faithful(_payload("req-a"), _CTX)  # consume only the first of two
    with pytest.raises(ProxyDivergence, match="never replayed"):
        replay.verify_fully_consumed()

    # Consuming both leaves nothing unconsumed.
    full = ReplayingProxy(store, "ep")
    full.faithful(_payload("req-a"), _CTX)
    full.faithful(_payload("req-b"), _CTX)
    assert full.unconsumed() == ()
    full.verify_fully_consumed()  # does not raise


def test_store_failure_paths_are_clear() -> None:
    store = TraceStore()
    # Loading an episode that was never opened: a not-found error (path names it).
    with pytest.raises(FileNotFoundError):
        store.load_episode("never-recorded")
    # Appending/closing an episode the store never opened is a usage error, loud.
    req = _payload("x")
    ev = SeamEvent(
        "ghost", 0, Seam.FUSE_TO_DECIDE, 0, 0.0, req, req, None, {}, canonicalize(req).digest, 0.0
    )
    with pytest.raises(KeyError):
        store.append_event("ghost", ev)
    with pytest.raises(KeyError):
        store.close_episode("ghost")


def test_non_openai_image_shapes_classify_as_sensor_to_caption() -> None:
    # The seam classifier must recognize vision content in every provider's encoding,
    # not just OpenAI's image_url (framework review, normalizer-coverage gap).
    from plumbline.adapters.om1 import OM1Adapter
    from plumbline.proxy.normalizers import contains_image

    gemini: JSONValue = {
        "contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": "aGk="}}]}]
    }
    gemini_snake: JSONValue = {"contents": [{"parts": [{"inline_data": {"data": "aGk="}}]}]}
    anthropic: JSONValue = {
        "messages": [{"content": [{"type": "image", "source": {"data": "aGk="}}]}]
    }
    text_only: JSONValue = {"messages": [{"content": [{"type": "text", "text": "no image here"}]}]}

    assert contains_image(gemini) and contains_image(gemini_snake) and contains_image(anthropic)
    assert not contains_image(text_only)

    adapter = OM1Adapter(proxy_base_url="http://localhost:8900")
    endpoint = "https://generativelanguage.googleapis.com/v1/models"
    assert adapter.seam_of(Payload(inline=gemini), endpoint) is Seam.SENSOR_TO_CAPTION
    assert adapter.seam_of(Payload(inline=anthropic), endpoint) is Seam.SENSOR_TO_CAPTION
    assert adapter.seam_of(Payload(inline=text_only), endpoint) is Seam.FUSE_TO_DECIDE
