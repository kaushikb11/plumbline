"""Matchers — input-consistency checks for counterfactual replay (eng spec §3.7).

FROZEN (CLAUDE.md invariant 1): `Matcher`, `MatchVerdict`, and the three built-in
matcher signatures are the contract. The method *bodies* are WS1 implementation.

The embedding matcher routes free text through a pinned `Embedder` so it is
deterministic and reproducible (§3.7). The default is a dependency-free
bag-of-tokens vectorizer; install a real pinned model with `set_embedder(...)`
(see `plumbline.embedding`). The frozen `EmbeddingMatcher(threshold)` signature
has no constructor injection point, so the pinned embedder lives at module scope
(flagged at the matcher).

The pin is held in a `contextvars.ContextVar`, NOT a plain module global, so that
concurrent episodes / parallel test threads that each pin a different embedder get
their OWN embedder instead of last-writer-wins cross-contamination. A plain module
global is process-wide: two async tasks or threads pinning different embedders
would silently both run under whichever wrote last. The ContextVar makes the pin
per-execution-context. Use `using_embedder(...)` to scope a pin to one
episode/replay, and `set_embedder(...)` for the single-threaded "pin once" case
(which still works exactly as before). `active_embedder()` / `active_embedder_name()`
surface which embedder actually ran (a core-only PASS silently used bag-of-tokens,
not a semantic model — these accessors make that observable).
"""

import contextlib
import contextvars
import math
from collections.abc import Callable, Iterator, Mapping
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
    # 0.0 = identical; the scale is matcher-specific and only comparable within a
    # fixed (seam, matcher) pair — do NOT compare distances across matchers/seams.
    distance: float
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
        verdict = _canonical_equal(live, recorded)
        if verdict is None:  # non-finite (NaN/Inf) can't be canonicalized
            return MatchVerdict(is_match=False, distance=1.0, reason="non-canonicalizable payload")
        if verdict:
            return MatchVerdict(is_match=True, distance=0.0, reason="exact canonical match")
        return MatchVerdict(is_match=False, distance=1.0, reason="structural mismatch")


@dataclass(frozen=True)
class EmbeddingMatcher:
    """Cosine distance over embeddings of free text (captions, prompts); match if
    distance < threshold (§3.7).

    The embedder can be pinned explicitly per matcher via the optional `embedder`
    field (recommended — the pinned model is then carried on the matcher and is
    reproducible/recordable, per §3.7). If unset, it falls back to the ambient
    context embedder (`using_embedder` / `set_embedder`) and finally the process
    default. Constructor injection is the additive field agreed as the deliberate
    core-interface change (docs/stability.md); the ambient hook remains for
    back-compat and for pinning across a whole run.
    """

    threshold: float
    embedder: "Embedder | None" = None

    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict:
        embed = self.embedder if self.embedder is not None else _current_embedder()
        live_vec = embed(_extract_text(live))
        recorded_vec = embed(_extract_text(recorded))
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
            equal = _canonical_equal(live, recorded)
            if equal is None:
                return MatchVerdict(False, 1.0, "non-canonicalizable payload")
            return MatchVerdict(equal, 0.0 if equal else 1.0, "no numeric fields; exact compare")
        max_diff = 0.0
        # Non-numeric structure (labels, keys, added/removed non-numeric fields) must
        # match exactly; only the numeric fields get tolerance. Otherwise a changed
        # `{"action": "stop"->"go"}` beside matching coordinates is a vacuous match.
        structural = _non_numeric_fields(live) != _non_numeric_fields(recorded)
        numeric_mismatch = False
        for key in keys:
            if key not in live_nums or key not in recorded_nums:
                structural = True
                continue
            a, b = live_nums[key], recorded_nums[key]
            if not math.isfinite(a) or not math.isfinite(b):
                # A NaN/Inf field is maximal corruption, not a zero-distance match —
                # keep the distance scale consistent with ExactMatcher's NaN handling.
                structural = True
                numeric_mismatch = True
                continue
            max_diff = max(max_diff, abs(a - b))
            if not math.isclose(a, b, rel_tol=self.rtol, abs_tol=self.atol):
                numeric_mismatch = True
        is_match = not structural and not numeric_mismatch
        # A structural mismatch is not a small numeric distance — report it as maximal
        # so the gate never reads a missing field as a near-identical match.
        distance = max(max_diff, 1.0) if structural else max_diff
        reason = (
            "structural mismatch: non-numeric fields or field set differ"
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


# The process-wide default embedder. This module attribute is the fallback used
# when no per-context pin is active, and it is kept in sync by `set_embedder` so
# that monkeypatching `plumbline.core.matcher._embed` (the interpreted injection
# point, §3.7) still swaps the embedder the matcher resolves.
_embed: Embedder = _bag_of_tokens

# The per-execution-context pin. Unset by default (LookupError falls back to the
# module-level `_embed`), so concurrent async tasks / threads that each pin their
# own embedder are isolated instead of clobbering one shared global. See the module
# docstring for why this is a ContextVar and not a plain global.
_embedder_var: contextvars.ContextVar[Embedder] = contextvars.ContextVar("plumbline_embedder")


def _current_embedder() -> Embedder:
    """Resolve the embedder active in THIS execution context.

    A per-context pin (from `set_embedder` / `using_embedder` in the current
    context) wins; otherwise fall back to the module-level `_embed` default. This
    is the single read path — `EmbeddingMatcher.matches` calls it at match time.
    """
    try:
        return _embedder_var.get()
    except LookupError:
        return _embed


def set_embedder(embedder: Embedder) -> None:
    """Pin `embedder` as the embedder for EmbeddingMatcher (§3.7).

    Pins it for the CURRENT execution context so a recorded episode and its replay
    use the same embedder — record the model's identity alongside the episode for
    reproducibility. Because the pin is context-local, concurrent episodes / test
    threads that each call `set_embedder` no longer cross-contaminate (previously
    the last writer's embedder silently won for everyone).

    The module-level `_embed` default is updated too, preserving the historical
    single-threaded contract (pin once, then use) and the monkeypatch hook. For a
    scoped pin that auto-restores, prefer `using_embedder`.
    """
    global _embed
    _embed = embedder
    _embedder_var.set(embedder)


@contextlib.contextmanager
def using_embedder(embedder: Embedder) -> Iterator[None]:
    """Pin `embedder` for the duration of a `with` block, then restore (§3.7).

    The clean way to pin one episode/replay's embedder without leaking it to other
    contexts: the pin is context-local and is undone via the ContextVar token on
    exit, even on exception. Does NOT touch the module-level `_embed` default, so
    unlike `set_embedder` it leaves no residue behind.
    """
    token = _embedder_var.set(embedder)
    try:
        yield
    finally:
        _embedder_var.reset(token)


def active_embedder() -> Embedder:
    """The embedder that would run right now in this context (pin or default).

    Lets callers surface which embedder actually served a verdict — e.g. to record
    it alongside an episode, or to flag that a core-only run silently used the
    dependency-free bag-of-tokens stand-in rather than a pinned semantic model.
    """
    return _current_embedder()


def active_embedder_name(embedder: Embedder | None = None) -> str:
    """A human-readable identity for `embedder` (defaults to the active one).

    Surfaces `bag_of_tokens` for the default so a PASS that never installed a
    semantic embedder is legible rather than silently trusted.
    """
    target = active_embedder() if embedder is None else embedder
    name = getattr(target, "__name__", "")
    return name if name else type(target).__name__


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


_Path = tuple[str | int, ...]


def _numeric_fields(payload: Payload) -> dict[_Path, float]:
    """Every numeric leaf of inline content, keyed by its path.

    Recurses into nested dicts/lists (`pose.x`, `commands[0].vx`) so coordinate
    payloads — which are almost always nested or list-shaped — are actually
    compared, rather than yielding an empty set and a vacuous match. Paths are
    TUPLES (not joined strings) so they are injective: `{"a": {"b": 5}}` and
    `{"a.b": 5}` map to distinct keys and cannot falsely compare equal.
    """
    result: dict[_Path, float] = {}

    def walk(prefix: _Path, value: JSONValue) -> None:
        if isinstance(value, bool):
            return  # bool is an int subclass; not a pose coordinate
        if isinstance(value, (int, float)):
            result[prefix] = float(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk((*prefix, key), item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk((*prefix, index), item)

    walk((), payload.inline)
    return result


def _canonical_equal(live: Payload, recorded: Payload) -> bool | None:
    """Whether two payloads have equal canonical digests. Returns None if either
    cannot be canonicalized (a non-finite float), so callers report a mismatch
    rather than letting the ValueError crash replay (invariant 5)."""
    try:
        return canonicalize(live).digest == canonicalize(recorded).digest
    except ValueError:
        return None


def _non_numeric_fields(payload: Payload) -> dict[_Path, str]:
    """Every NON-numeric leaf (str/bool/null) keyed by (injective, tuple) path,
    tagged by type, so the structural skeleton is compared exactly (numeric leaves
    get tolerance instead)."""
    result: dict[_Path, str] = {}

    def walk(prefix: _Path, value: JSONValue) -> None:
        if isinstance(value, bool):
            result[prefix] = f"bool:{value}"
        elif isinstance(value, (int, float)):
            return  # numeric: compared with tolerance elsewhere
        elif isinstance(value, dict):
            for key, item in value.items():
                walk((*prefix, key), item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk((*prefix, index), item)
        else:  # str or None
            result[prefix] = f"{type(value).__name__}:{value}"

    walk((), payload.inline)
    return result
