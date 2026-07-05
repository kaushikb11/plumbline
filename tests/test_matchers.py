"""Matcher tests (eng spec §15, §3.7).

Exact, embedding, and tolerance matchers against crafted near-miss payloads. The
embedding case injects a deterministic mock embedder so the verdict is
reproducible (§3.7 requires the real matcher's embedder to be pinned/recorded).
"""

from collections.abc import Mapping

import pytest
from plumbline.core.matcher import EmbeddingMatcher, ExactMatcher, NumericToleranceMatcher
from plumbline.core.trace import Payload


def deterministic_embed(text: str) -> Mapping[str, float]:
    """A pinned, deterministic bag-of-tokens embedder for tests (§3.7).

    Returns a sparse term-count vector; case- and whitespace-insensitive, so
    formatting noise collapses to distance 0 while disjoint vocabulary is far.
    """
    counts: dict[str, float] = {}
    for token in text.lower().split():
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def test_exact_matcher_on_near_miss_action_payloads() -> None:
    matcher = ExactMatcher()
    base = Payload(inline={"action": "move", "x": 0.2, "yaw": 0.5})
    identical = Payload(inline={"action": "move", "x": 0.2, "yaw": 0.5})
    near_miss = Payload(inline={"action": "move", "x": 0.2, "yaw": 0.4})

    assert matcher.matches(base, identical).is_match is True  # <-- NotImplementedError now
    assert matcher.matches(base, identical).distance == 0.0

    verdict = matcher.matches(base, near_miss)
    assert verdict.is_match is False
    assert verdict.distance > 0.0


def test_embedding_matcher_uses_pinned_deterministic_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject the deterministic embedder at the expected module-level hook.
    # NOTE: `_embed` is the interpreted injection point — the frozen
    # EmbeddingMatcher(threshold) signature has no constructor for an embedder
    # (§3.7), so a module-level pinned function is the assumed mechanism. Revisit
    # when the matcher is implemented.
    monkeypatch.setattr("plumbline.core.matcher._embed", deterministic_embed, raising=False)
    matcher = EmbeddingMatcher(threshold=0.25)

    base = Payload(inline={"caption": "human 0.3 m to your left"})
    formatting_noise = Payload(inline={"caption": "HUMAN  0.3 m   to your LEFT"})
    divergent = Payload(inline={"caption": "the path ahead is clear"})

    assert matcher.matches(base, formatting_noise).is_match is True  # <-- NotImplementedError now
    assert matcher.matches(base, divergent).is_match is False


def test_numeric_tolerance_matcher_on_pose_payloads() -> None:
    matcher = NumericToleranceMatcher(rtol=1e-3, atol=1e-3)
    base = Payload(inline={"x": 1.000, "y": 2.000, "yaw": 0.500})
    within_tolerance = Payload(inline={"x": 1.0005, "y": 1.9995, "yaw": 0.5004})
    outside_tolerance = Payload(inline={"x": 1.500, "y": 2.000, "yaw": 0.500})

    assert matcher.matches(base, within_tolerance).is_match is True  # <-- NotImplementedError now
    assert matcher.matches(base, outside_tolerance).is_match is False


# --- ActionSchemaMatcher (§9.1, §14.6) --------------------------------------

from plumbline.adapters import (  # noqa: E402
    ActionSchemaMatcher,
    GenericActionSchema,
    recommended_behavior_matcher,
)
from plumbline.core.matcher import ExactMatcher as _Exact  # noqa: E402


def _tool(name: str, arguments: str) -> Payload:
    """A tool-call action payload (the shape a function-calling runtime emits)."""
    return Payload(inline={"tool_calls": [{"function": {"name": name, "arguments": arguments}}]})


def _speed(value: float) -> Payload:
    return _tool("move_forward", f'{{"speed": {value}}}')


def test_action_schema_matcher_arg_within_tolerance_matches() -> None:
    matcher = ActionSchemaMatcher(GenericActionSchema(), atol=1e-2)
    live, recorded = _speed(0.301), _speed(0.30)
    assert matcher.matches(live, recorded).is_match  # within tolerance
    assert matcher.matches(live, recorded).distance == 0.0
    assert not _Exact().matches(live, recorded).is_match  # ...where ExactMatcher fails


def test_action_schema_matcher_arg_outside_tolerance_fails() -> None:
    matcher = ActionSchemaMatcher(GenericActionSchema(), atol=1e-2)
    verdict = matcher.matches(_speed(0.50), _speed(0.30))
    assert not verdict.is_match
    assert verdict.distance == 1.0  # single action, one mismatch


def test_action_schema_matcher_changed_name_fails() -> None:
    matcher = ActionSchemaMatcher(GenericActionSchema())
    assert not matcher.matches(_tool("turn_left", "{}"), _tool("turn_right", "{}")).is_match


def test_action_schema_matcher_dropped_action_is_half() -> None:
    matcher = ActionSchemaMatcher(GenericActionSchema())
    two = Payload(
        inline={
            "tool_calls": [
                {"function": {"name": "move_forward", "arguments": "{}"}},
                {"function": {"name": "speak", "arguments": '{"text": "hi"}'}},
            ]
        }
    )
    one = _tool("move_forward", "{}")
    verdict = matcher.matches(one, two)
    assert not verdict.is_match
    assert verdict.distance == 0.5  # length gap 1 over max length 2


def test_action_schema_matcher_non_numeric_arg_differs_fails() -> None:
    matcher = ActionSchemaMatcher(GenericActionSchema())
    assert not matcher.matches(
        _tool("speak", '{"text": "hello"}'), _tool("speak", '{"text": "x"}')
    ).is_match


def test_action_schema_matcher_both_unparseable_match() -> None:
    # §14.6 open default: two payloads that parse to no actions read as equivalent
    # (consistent with structural_equivalence's both-empty convention).
    matcher = ActionSchemaMatcher(GenericActionSchema())
    verdict = matcher.matches(Payload(inline={"nope": 1}), Payload(inline={"other": 2}))
    assert verdict.is_match
    assert verdict.distance == 0.0


def test_action_schema_matcher_reorder_tolerance() -> None:
    schema = GenericActionSchema()
    plan_ab = Payload(
        inline={
            "tool_calls": [
                {"function": {"name": "move_forward", "arguments": '{"speed": 0.3}'}},
                {"function": {"name": "turn_left", "arguments": '{"rate": 0.1}'}},
            ]
        }
    )
    plan_ba = Payload(
        inline={
            "tool_calls": [
                {"function": {"name": "turn_left", "arguments": '{"rate": 0.1}'}},
                {"function": {"name": "move_forward", "arguments": '{"speed": 0.3}'}},
            ]
        }
    )
    assert not ActionSchemaMatcher(schema).matches(plan_ab, plan_ba).is_match  # order-sensitive
    assert recommended_behavior_matcher(schema).matches(plan_ab, plan_ba).is_match  # reorder-ok
    # A genuinely different multiset still mismatches, even reorder-insensitive.
    different = Payload(
        inline={
            "tool_calls": [
                {"function": {"name": "turn_left", "arguments": '{"rate": 0.1}'}},
                {"function": {"name": "back_up", "arguments": '{"speed": 0.2}'}},
            ]
        }
    )
    assert not recommended_behavior_matcher(schema).matches(plan_ab, different).is_match


# --- pinned-embedder mechanism: context isolation (§3.7) --------------------

import contextvars  # noqa: E402
import threading  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import plumbline.core.matcher as matcher_module  # noqa: E402
from plumbline.core.matcher import (  # noqa: E402
    Embedder,
    active_embedder,
    active_embedder_name,
    set_embedder,
    using_embedder,
)


@contextmanager
def _restore_module_default() -> Iterator[None]:
    """Save/restore the process-wide `_embed` default so a test that calls
    `set_embedder` (which syncs the module default for back-compat) does not leak
    its embedder into sibling tests."""
    original = matcher_module._embed
    try:
        yield
    finally:
        matcher_module._embed = original


def _const_embedder(_text: str) -> Mapping[str, float]:
    """Everything maps to the same vector -> cosine distance 0 -> always matches."""
    return {"const": 1.0}


def _per_text_embedder(text: str) -> Mapping[str, float]:
    """Bag-of-tokens -> disjoint captions are orthogonal -> distance 1 -> no match."""
    return {token: 1.0 for token in text.lower().split()}


def test_using_embedder_scopes_the_pin_and_restores() -> None:
    # Outside the block: the module default (dependency-free bag-of-tokens).
    assert active_embedder_name() == "_bag_of_tokens"
    with using_embedder(_const_embedder):
        assert active_embedder() is _const_embedder
    # Restored on exit — no leak past the block.
    assert active_embedder_name() == "_bag_of_tokens"


def test_concurrent_threads_get_isolated_embedders() -> None:
    """Two threads each `set_embedder` to a DIFFERENT embedder and must each see
    their OWN — not last-writer-wins. This is the production MAJOR: with a plain
    process-global, both threads would run under whichever wrote last.

    A plain `threading.Thread` does NOT auto-copy the parent context (each thread
    starts with a fresh context), so setting the pin *inside* each worker gives it
    an isolated ContextVar. A barrier forces both writes to land before either
    thread reads, so a global last-writer-wins would be observable here.
    """
    base = Payload(inline={"caption": "human left"})
    divergent = Payload(inline={"caption": "path clear"})
    barrier = threading.Barrier(2)
    results: dict[str, bool] = {}

    def worker(tag: str, embedder: Embedder, expect_match: bool) -> None:
        set_embedder(embedder)  # set WITHIN the thread -> this thread's context only
        barrier.wait()  # both threads have now pinned; force interleaving
        matcher = EmbeddingMatcher(threshold=0.25)
        own = active_embedder() is embedder
        verdict = matcher.matches(base, divergent).is_match is expect_match
        results[tag] = own and verdict

    with _restore_module_default():  # set_embedder syncs the shared module default; undo it
        threads = [
            threading.Thread(target=worker, args=("const", _const_embedder, True)),
            threading.Thread(target=worker, args=("per_text", _per_text_embedder, False)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # Each thread saw its OWN embedder and thus its OWN verdict, despite the other
    # thread concurrently pinning a different one.
    assert results == {"const": True, "per_text": True}


def test_set_embedder_single_threaded_pin_then_use_unchanged() -> None:
    """The historical contract: `set_embedder` then use, in the same context,
    still pins for subsequent matches (existing single-threaded behavior).

    Run in a copied context so the per-context pin does not leak into the shared
    main context (which would shadow sibling tests that monkeypatch the module
    default); the module default is saved/restored separately.
    """
    base = Payload(inline={"caption": "human left"})
    divergent = Payload(inline={"caption": "path clear"})

    def body() -> None:
        set_embedder(_const_embedder)
        assert active_embedder() is _const_embedder
        assert matcher_module._embed is _const_embedder  # module default kept in sync
        assert EmbeddingMatcher(threshold=0.25).matches(base, divergent).is_match is True

    with _restore_module_default():
        contextvars.copy_context().run(body)


def test_per_instance_embedder_wins_over_ambient() -> None:
    # The signed-off additive field: a matcher carrying its own embedder ignores the
    # ambient (using_embedder / set_embedder) pin — explicit and recordable per matcher.
    a = Payload(inline={"caption": "hello world"})
    b = Payload(inline={"caption": "goodbye moon"})
    with using_embedder(_per_text_embedder):  # ambient distinguishes the two texts
        assert EmbeddingMatcher(0.2).matches(a, b).is_match is False  # uses ambient
        # own embedder (_const_embedder -> identical vectors) overrides the ambient one
        assert EmbeddingMatcher(0.2, embedder=_const_embedder).matches(a, b).is_match is True
