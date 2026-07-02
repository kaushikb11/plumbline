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
