"""The developer-experience surface: flat public imports, the make_seam_event
helper, the typed exceptions (which stay backward-compatible with the legacy
KeyError/FileNotFoundError catches), and the adapter conformance check.
"""

import importlib

import pytest
from plumbline import (
    DigestMismatch,
    Payload,
    Recorder,
    Seam,
    SeamEvent,
    TraceStore,
    VirtualClock,
    canonicalize,
    make_seam_event,
)
from plumbline.core.store import EpisodeNotFound, EpisodeNotOpen


def test_top_level_flat_imports_and_version() -> None:
    plumbline = importlib.import_module("plumbline")
    # The essentials are reachable without deep imports.
    for name in ("Seam", "SeamEvent", "make_seam_event", "Recorder", "Replayer", "TraceStore"):
        assert name in plumbline.__all__
        assert hasattr(plumbline, name)
    assert isinstance(plumbline.__version__, str) and plumbline.__version__


def test_make_seam_event_computes_digest_and_defaults() -> None:
    req = Payload(inline={"input": "frame"})
    event = make_seam_event(
        episode_id="ep",
        seq=0,
        seam=Seam.SENSOR_TO_CAPTION,
        logical_tick=0,
        request=req,
        response=Payload(inline={"output": "caption"}),
    )
    assert event.request_digest == canonicalize(req).digest  # auto-computed
    assert event.wall_ts == 0.0 and event.latency_ms == 0.0 and event.params == {}
    # It records cleanly (the digest is correct by construction).
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep", {})
    recorder.record(event)
    recorder.close_episode("ep")
    assert len(store.load_episode("ep").events) == 1


def test_wrong_digest_is_rejected_at_record_time() -> None:
    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    recorder.open_episode("ep", {})
    bad = SeamEvent(
        episode_id="ep",
        seq=0,
        seam=Seam.SENSOR_TO_CAPTION,
        logical_tick=0,
        wall_ts=0.0,
        request=Payload(inline={"input": "frame"}),
        response=Payload(inline={"output": "caption"}),
        model_id=None,
        params={},
        request_digest="deadbeef",  # does not match the request
        latency_ms=0.0,
    )
    with pytest.raises(DigestMismatch):
        recorder.record(bad)


def test_typed_exceptions_stay_backward_compatible() -> None:
    # EpisodeNotOpen is-a KeyError; EpisodeNotFound is-a FileNotFoundError, so existing
    # `except`/`pytest.raises` clauses (and the server's 404 mapping) keep working.
    assert issubclass(EpisodeNotOpen, KeyError)
    assert issubclass(EpisodeNotFound, FileNotFoundError)

    store = TraceStore()
    recorder = Recorder(store, VirtualClock())
    event = make_seam_event(
        episode_id="never-opened",
        seq=0,
        seam=Seam.SENSOR_TO_CAPTION,
        logical_tick=0,
        request=Payload(inline={"input": "x"}),
        response=Payload(inline={"output": "y"}),
    )
    with pytest.raises(KeyError):  # still catchable as the legacy type
        recorder.record(event)

    with pytest.raises(FileNotFoundError):
        store.load_episode("does-not-exist")


def test_adapter_conformance_check() -> None:
    from plumbline.adapters import ConformanceError, assert_conforms
    from plumbline.adapters._template import TemplateAdapter

    assert_conforms(TemplateAdapter("http://localhost"))  # the skeleton conforms

    class Broken:
        pass  # implements none of the contract

    with pytest.raises((ConformanceError, TypeError, AssertionError)):
        assert_conforms(Broken())


def test_toy_loop_example_runs() -> None:
    # The zero-setup on-ramp actually works end-to-end.
    toy = importlib.import_module("examples.toy_loop")
    toy.main()
