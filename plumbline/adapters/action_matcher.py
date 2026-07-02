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

    def matches(self, live: Payload, recorded: Payload) -> MatchVerdict:
        live_actions = self._parse(live)
        recorded_actions = self._parse(recorded)
        total = max(len(live_actions), len(recorded_actions))
        if total == 0:
            # Both plans empty -> match, consistent with structural_equivalence's
            # both-empty convention (§14.6 open; documented, not silently inverted).
            return MatchVerdict(is_match=True, distance=0.0, reason="both action plans empty")
        arg_matcher = NumericToleranceMatcher(rtol=self.rtol, atol=self.atol)
        paired = min(len(live_actions), len(recorded_actions))
        mismatches = sum(
            not _action_equal(live_actions[i], recorded_actions[i], arg_matcher)
            for i in range(paired)
        )
        length_gap = abs(len(live_actions) - len(recorded_actions))
        # (mismatched aligned actions + length gap) / max length, bounded [0, 1] —
        # the §8.3/§14.6 shape, applied within one plan of Actions.
        distance = (mismatches + length_gap) / total
        return MatchVerdict(
            is_match=distance == 0.0,
            distance=distance,
            reason=f"{mismatches}/{paired} aligned actions differ; length gap {length_gap}",
        )

    def _parse(self, payload: Payload) -> tuple[Action, ...]:
        try:
            return tuple(self.schema.parse(payload))
        except Exception:  # a third-party schema that raises -> treat as unparseable
            return ()


def _action_equal(live: Action, recorded: Action, arg_matcher: NumericToleranceMatcher) -> bool:
    if live.kind != recorded.kind or live.name != recorded.name:
        return False
    return arg_matcher.matches(
        Payload(inline=dict(live.args)), Payload(inline=dict(recorded.args))
    ).is_match
