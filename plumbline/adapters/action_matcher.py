"""ActionSchema-derived behavior matcher (engineering spec §9.1, §14.6).

The gate scores behavioral drift by comparing DECIDE_TO_ACT action-plan payloads
with a `Matcher`. The exact / embedding / tolerance matchers work on raw JSON; this
one works on TYPED actions: it parses both payloads via an `ActionSchema` and
compares the resulting `Action` tuples position-by-position — same kind and name,
numeric args within tolerance — so `move(x=0.30)` vs `move(x=0.301)` is a match
while a changed skill or a dropped action is not. It is §14.6's "ActionSchema-
derived matcher" open choice made concrete; inject it as `GateSpec.behavior_matcher`.

Structurally satisfies the frozen core `Matcher` protocol (`matches -> MatchVerdict`)
and lives in `adapters/` because it depends on the (non-frozen) `ActionSchema`
surface — `core/` is frozen and must not depend on the adapter layer.
"""

from dataclasses import dataclass

from plumbline.adapters.base import Action, ActionSchema
from plumbline.core.matcher import MatchVerdict, NumericToleranceMatcher
from plumbline.core.trace import Payload


@dataclass(frozen=True)
class ActionSchemaMatcher:
    """Compare two action-plan payloads as typed `Action`s, with per-arg numeric
    tolerance (§14.6). Delegates arg comparison to `NumericToleranceMatcher`, so it
    inherits tolerance for numeric args, exact compare for non-numeric args, and
    NaN handling — consistent with the existing tolerance semantics."""

    schema: ActionSchema
    rtol: float = 1e-3
    atol: float = 1e-3
    # order_sensitive=False compares plans as multisets, so a behaviorally-identical
    # reordering isn't drift (§14.6 open choice — some runtimes are order-dependent).
    order_sensitive: bool = True

    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict:
        live_actions = self._parse(live)
        recorded_actions = self._parse(recorded)
        total = max(len(live_actions), len(recorded_actions))
        if total == 0:
            # Both plans empty -> match, consistent with structural_equivalence's
            # both-empty convention (§14.6 open; documented, not silently inverted).
            return MatchVerdict(is_match=True, distance=0.0, reason="both action plans empty")
        arg_matcher = NumericToleranceMatcher(rtol=self.rtol, atol=self.atol)
        if self.order_sensitive:
            paired = min(len(live_actions), len(recorded_actions))
            matched = sum(
                _action_equal(live_actions[i], recorded_actions[i], arg_matcher)
                for i in range(paired)
            )
        else:
            matched = _multiset_matched(live_actions, recorded_actions, arg_matcher)
        # (total - matched) / max length, bounded [0, 1] — the §8.3/§14.6 shape (a
        # dropped/added/changed action still counts; a reorder counts only when
        # order_sensitive).
        distance = (total - matched) / total
        return MatchVerdict(
            is_match=distance == 0.0,
            distance=distance,
            reason=f"{matched}/{total} actions matched (order_sensitive={self.order_sensitive})",
        )

    def _parse(self, payload: Payload) -> tuple[Action, ...]:
        try:
            return tuple(self.schema.parse(payload))
        except Exception:  # a third-party schema that raises -> treat as unparseable
            return ()


def recommended_behavior_matcher(
    schema: ActionSchema, *, rtol: float = 1e-3, atol: float = 1e-3
) -> ActionSchemaMatcher:
    """The recommended gate `behavior_matcher` (§14.6): typed, numeric-tolerant, and
    reorder-insensitive — so it doesn't cry wolf on jitter or benign reordering.
    ExactMatcher stays available; the matcher choice is a human decision, so this is a
    recommendation, not a silent default change."""
    return ActionSchemaMatcher(schema, rtol=rtol, atol=atol, order_sensitive=False)


def _multiset_matched(
    live: tuple[Action, ...], recorded: tuple[Action, ...], arg_matcher: NumericToleranceMatcher
) -> int:
    """Greedy multiset match: each live action consumes the first unused recorded
    action it equals. (Tolerance-equality isn't transitive, so this is conservative —
    same spirit as structural_equivalence's alignment.)"""
    remaining = list(recorded)
    matched = 0
    for action in live:
        for index, candidate in enumerate(remaining):
            if _action_equal(action, candidate, arg_matcher):
                del remaining[index]
                matched += 1
                break
    return matched


def _action_equal(live: Action, recorded: Action, arg_matcher: NumericToleranceMatcher) -> bool:
    if live.kind != recorded.kind or live.name != recorded.name:
        return False
    return arg_matcher.matches(
        Payload(inline=dict(live.args)), Payload(inline=dict(recorded.args))
    ).is_match
