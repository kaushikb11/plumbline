"""Noise-floor calibration test (eng spec §15, §7.2).

With a fixed input and a known-temperature stub decider, the decision-stability
floor sigma must converge to the analytic self-divergence as N grows: two i.i.d.
halves of the same distribution agree in the limit, so sigma -> 0 and scales as
~1/sqrt(N). This proves the floor measures what it claims, so a fidelity gap is
only credited beyond it (§7.2).

The Protocol below pins the expected signature (§7.1: decision_distribution ->
the stability floor derived from it).
"""

import math
import random
from collections.abc import Mapping
from typing import Protocol, cast

import pytest
from plumbline.core.trace import JSONValue

from tests.toyloop import DEFAULT_RULES, StubDecider, fuse, load_unimplemented

# Sample budgets, strictly increasing, spanning ~64x so the 1/sqrt(N) law is visible.
_SAMPLE_SIZES: tuple[int, ...] = (64, 256, 1024, 4096)


class _DecisionStabilityFn(Protocol):
    """sigma(x): expected divergence between two halves of N decision samples (§7.2)."""

    def __call__(
        self,
        decider: "_DeciderFn",
        context: str,
        n: int,
    ) -> float: ...


class _DeciderFn(Protocol):
    def __call__(self, prompt: str) -> Mapping[str, JSONValue]: ...


def test_sigma_converges_to_analytic_self_divergence() -> None:
    decider = StubDecider(rng=random.Random(0), temperature=0.3)
    # A fixed input: a fused prompt carrying a known close-obstacle distance.
    context = fuse(["obstacle 0.30 m at 0 deg"], DEFAULT_RULES)

    decision_stability = cast(
        _DecisionStabilityFn,
        load_unimplemented("plumbline.fidelity", "decision_stability"),  # AttributeError now
    )

    sigmas = [decision_stability(decider.decide, context, n) for n in _SAMPLE_SIZES]

    # Monotone non-increasing: more samples, tighter floor. (Pair consecutive
    # sigmas; both slices have the same length so strict zip is well-formed.)
    assert all(later <= earlier for earlier, later in zip(sigmas[:-1], sigmas[1:], strict=True))
    # Converging toward the analytic limit of 0 (self-divergence of one distribution).
    assert sigmas[-1] < sigmas[0]
    # Sampling noise scales as 1/sqrt(N): the floor shrinks by ~sqrt(64x) = 8x.
    expected_ratio = math.sqrt(_SAMPLE_SIZES[-1] / _SAMPLE_SIZES[0])
    assert sigmas[0] / sigmas[-1] == pytest.approx(expected_ratio, rel=0.5)
