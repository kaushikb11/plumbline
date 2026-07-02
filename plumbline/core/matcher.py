"""Matchers — input-consistency checks for counterfactual replay (eng spec §3.7).

FROZEN (CLAUDE.md invariant 1): `Matcher`, `MatchVerdict`, and the three built-in
matcher signatures are the contract. The method *bodies* are WS1 implementation.

The embedding matcher routes free text through a module-level pinned `Embedder`
so it is deterministic and reproducible (§3.7). The default is a dependency-free
bag-of-tokens vectorizer; install a real pinned model with `set_embedder(...)`
(see `plumbline.embedding`). The frozen `EmbeddingMatcher(threshold)` signature
has no constructor injection point, so the module-level hook is the mechanism
(flagged at the matcher).
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from plumbline.core.trace import JSONValue, Payload, canonicalize

# A pinned text embedder: text -> a (sparse or index-keyed dense) vector. Cosine
# distance over two embeddings drives the free-text matcher (§3.7).
Embedder = Callable[[str], Mapping[str, float]]


@dataclass(frozen=True)
class MatchVerdict:
    """Verdict of comparing a live request against a recorded one (§3.7).

    NOTE: §3.7 shows this as a plain `@dataclass`; it is frozen here under the
    frozen-data invariant (CLAUDE.md). Pure data, so the strengthening is safe.
    """

    is_match: bool
    distance: float  # 0.0 = identical; matcher-specific scale
    reason: str


class Matcher(Protocol):
    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict: ...


@dataclass(frozen=True)
class ExactMatcher:
    """Byte/structural equality for structured fields (action schemas, params).

    Compares canonical serializations, not Python `==`, so JSON-distinct values
    that Python conflates (`1 == 1.0 == True`) are treated as the mismatches the
    "byte equality" contract implies and that the store preserves on disk.
    """

    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict:
        if canonicalize(live).digest == canonicalize(recorded).digest:
            return MatchVerdict(is_match=True, distance=0.0, reason="exact canonical match")
        return MatchVerdict(is_match=False, distance=1.0, reason="structural mismatch")


@dataclass(frozen=True)
class EmbeddingMatcher:
    """Cosine distance over embeddings of free text (captions, prompts); match if
    distance < threshold (§3.7).

    NOTE: §3.7 says the embedding model is itself pinned and recorded so the
    matcher is reproducible, but the spec signature exposes only `threshold`; the
    pinned model is the module-level `_embed` hook (flagged in the module
    docstring), not a constructor argument.
    """

    threshold: float

    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict:
        live_vec = _embed(_extract_text(live))
        recorded_vec = _embed(_extract_text(recorded))
        distance = _cosine_distance(live_vec, recorded_vec)
        is_match = distance < self.threshold
        return MatchVerdict(
            is_match=is_match,
            distance=distance,
            reason=f"embedding cosine distance {distance:.4f} "
            f"{'<' if is_match else '>='} threshold {self.threshold}",
        )


@dataclass(frozen=True)
class NumericToleranceMatcher:
    """Tolerance comparison for pose/coordinate payloads (§3.7)."""

    rtol: float
    atol: float

    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict:
        live_nums = _numeric_fields(live)
        recorded_nums = _numeric_fields(recorded)
        keys = live_nums.keys() | recorded_nums.keys()
        if not keys:
            # No comparable numeric fields anywhere: this matcher can't measure a
            # tolerance, so fall back to exact equality rather than a vacuous match.
            equal = canonicalize(live).digest == canonicalize(recorded).digest
            return MatchVerdict(equal, 0.0 if equal else 1.0, "no numeric fields; exact compare")
        max_diff = 0.0
        structural = False  # a field present on only one side
        numeric_mismatch = False
        for key in keys:
            if key not in live_nums or key not in recorded_nums:
                structural = True
                continue
            a, b = live_nums[key], recorded_nums[key]
            max_diff = max(max_diff, abs(a - b))
            if not math.isclose(a, b, rel_tol=self.rtol, abs_tol=self.atol):
                numeric_mismatch = True
        is_match = not structural and not numeric_mismatch
        # A structural mismatch is not a small numeric distance — report it as maximal
        # so the gate never reads a missing field as a near-identical match.
        distance = max(max_diff, 1.0) if structural else max_diff
        reason = (
            "structural mismatch: numeric field present on only one side"
            if structural
            else f"max abs field difference {max_diff:.6g}"
        )
        return MatchVerdict(is_match=is_match, distance=distance, reason=reason)


# --- pinned embedder + extraction helpers (implementation detail) -----------


def _bag_of_tokens(text: str) -> Mapping[str, float]:
    """The dependency-free default embedder: a deterministic bag-of-tokens vector.

    Case- and whitespace-insensitive term counts. A crude but pinned stand-in;
    `set_embedder` swaps in a real semantic model (`plumbline.embedding`).
    """
    counts: dict[str, float] = {}
    for token in text.lower().split():
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


# The currently pinned embedder. EmbeddingMatcher.matches resolves this at call
# time, so `set_embedder` (or monkeypatching `_embed`) swaps it globally.
_embed: Embedder = _bag_of_tokens


def set_embedder(embedder: Embedder) -> None:
    """Install `embedder` as the pinned embedder for EmbeddingMatcher (§3.7).

    Pins it globally so a recorded episode and its replay use the same embedder —
    record the model's identity alongside the episode for reproducibility.
    """
    global _embed
    _embed = embedder


def _cosine_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 and norm_b == 0.0:
        return 0.0  # both empty -> identical, not maximally diverged
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def _extract_text(payload: Payload) -> str:
    """Concatenate every string leaf of the payload's inline content."""
    parts: list[str] = []

    def walk(value: JSONValue) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(payload.inline)
    return " ".join(parts)


def _numeric_fields(payload: Payload) -> dict[str, float]:
    """Every numeric leaf of inline content, keyed by its path.

    Recurses into nested dicts/lists (`pose.x`, `commands[0].vx`) so coordinate
    payloads — which are almost always nested or list-shaped — are actually
    compared, rather than yielding an empty set and a vacuous match.
    """
    result: dict[str, float] = {}

    def walk(prefix: str, value: JSONValue) -> None:
        if isinstance(value, bool):
            return  # bool is an int subclass; not a pose coordinate
        if isinstance(value, (int, float)):
            result[prefix] = float(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(f"{prefix}[{index}]", item)

    walk("", payload.inline)
    return result
