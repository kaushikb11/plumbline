"""Behavioral-equivalence judge (engineering spec §7.5).

When ground truth is unavailable (real-robot recordings), fidelity and regression
comparisons fall back to behavioral equivalence between two runs' action
sequences. Two mechanisms:

  * Structural — typed action plans compared field-wise via a `Matcher`
    (ExactMatcher / NumericToleranceMatcher over the action schema).
  * Semantic — an LLM-as-judge given both action sequences and asked whether they
    are behaviorally equivalent. The judge call is routed through the proxy as a
    `JudgeModel` callable, so it is recorded and replayed exactly like any other
    model call: the eval is as reproducible as the thing it evaluates. The judge's
    own noise floor is measured the same way as sigma (§7.2 split-half).

================================  HUMAN REVIEW  ================================
§14.6 (action equivalence) is OPEN and surfaced here, not hidden:

  * Structural alignment for sequences of differing length. This uses index-wise
    alignment and PENALIZES insertions/deletions (a longer or shorter candidate
    is not free). A proper edit-distance alignment is the open choice; the
    deliberate conservative default is that extra/missing actions count against
    equivalence so the metric cannot flatter.
  * How much to lean on the LLM judge vs the structural comparison. Both are
    provided; the caller decides. The LLM judge is itself nondeterministic, so its
    verdict must be reported against `judge_noise_floor` before it is trusted —
    exactly as fidelity numbers are reported against sigma.

REPLAY CAVEAT (flagged): faithful replay serves recorded responses by
request_digest, so N identical judge prompts collapse to one recorded response.
`judge_noise_floor` (which needs the *distribution* over repeated identical calls)
is therefore a record/live-mode measurement; reproducing it under replay needs
sequence-aware serving (a proxy enhancement), not by-digest serving.
===============================================================================
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.trace import JSONValue, Payload
from plumbline.fidelity.decision import Divergence, self_divergence, total_variation

# A judge model call routed through the proxy (record or replay).
JudgeModel = Callable[[Payload], Payload]

_EXACT_MATCHER: Matcher = ExactMatcher()
_WORD = re.compile(r"[a-z]+")
_NEGATION_WORDS = frozenset({"not", "no", "never", "cannot", "without"})
_DIFFERENT_WORDS = frozenset({"diverge", "diverges", "diverged", "differ", "differs", "different"})
_SAME_WORDS = frozenset({"equivalent", "identical", "same", "equal", "yes"})


@dataclass(frozen=True)
class JudgeVerdict:
    equivalent: bool
    distance: float  # 0.0 = identical behavior; method-specific scale up to 1.0
    reason: str
    method: str  # "structural" | "semantic"


def structural_equivalence(
    recorded: Sequence[Payload],
    candidate: Sequence[Payload],
    *,
    matcher: Matcher = _EXACT_MATCHER,
) -> JudgeVerdict:
    """Field-wise comparison of typed action plans (§7.5 structural).

    Aligns index-wise and penalizes length differences as insertions/deletions
    (§8.3, §14.6): distance = (mismatched aligned steps + length gap) / max length.
    """
    paired = min(len(recorded), len(candidate))
    mismatches = sum(not matcher.matches(candidate[i], recorded[i]).is_match for i in range(paired))
    length_gap = abs(len(recorded) - len(candidate))
    total_steps = max(len(recorded), len(candidate))
    if total_steps == 0:
        return JudgeVerdict(True, 0.0, "both sequences empty", "structural")
    distance = (mismatches + length_gap) / total_steps
    return JudgeVerdict(
        equivalent=distance == 0.0,
        distance=distance,
        reason=f"{mismatches}/{paired} aligned steps differ; length gap {length_gap}",
        method="structural",
    )


def behavioral_equivalence_prompt(
    sequence_a: Sequence[Payload], sequence_b: Sequence[Payload]
) -> Payload:
    """The judge prompt comparing two action sequences. Deterministic in its
    inputs, so faithful replay keys it by a stable request_digest."""
    return Payload(
        inline={
            "task": "Decide whether two robot action sequences are behaviorally equivalent.",
            "sequence_a": [event.inline for event in sequence_a],
            "sequence_b": [event.inline for event in sequence_b],
            "answer_format": "Reply 'EQUIVALENT' or 'NOT EQUIVALENT' with a brief reason.",
        }
    )


def semantic_equivalence(
    sequence_a: Sequence[Payload],
    sequence_b: Sequence[Payload],
    judge_model: JudgeModel,
    *,
    n: int = 1,
) -> JudgeVerdict:
    """LLM-as-judge behavioral equivalence (§7.5).

    `judge_model` is the proxy-routed model call, so the judgment is recorded and
    replayable. Sampling N>1 majority-votes; `distance` is the fraction voting
    'not equivalent'. Report against `judge_noise_floor` before trusting it.
    """
    prompt = behavioral_equivalence_prompt(sequence_a, sequence_b)
    equivalent_votes = sum(_parse_equivalent(judge_model(prompt)) for _ in range(n))
    distance = 1.0 - equivalent_votes / n
    return JudgeVerdict(
        # Strict majority; a tie breaks to NOT equivalent (conservative for a gate).
        equivalent=equivalent_votes * 2 > n,
        distance=distance,
        reason=f"LLM judge: {equivalent_votes}/{n} samples equivalent",
        method="semantic",
    )


def judge_noise_floor(
    sequence_a: Sequence[Payload],
    sequence_b: Sequence[Payload],
    judge_model: JudgeModel,
    n: int,
    *,
    divergence: Divergence = total_variation,
) -> float:
    """The LLM judge's own self-disagreement on a fixed pair (§7.5). Draws 2N and
    splits into two N-halves, matching `decision_stability`'s sample-size convention
    so the judge floor is measured at the same scale as sigma. Record/live-mode."""
    prompt = behavioral_equivalence_prompt(sequence_a, sequence_b)
    labels = [
        "equivalent" if _parse_equivalent(judge_model(prompt)) else "not_equivalent"
        for _ in range(2 * n)
    ]
    return self_divergence(labels, divergence=divergence)


def _parse_equivalent(response: Payload) -> bool:
    """Parse the judge's verdict — WORD- and CLAUSE-aware (not substring).

    The prompt enforces 'EQUIVALENT' / 'NOT EQUIVALENT', but a judge that ignores
    the format may hedge or compound. Polarity WORDS are matched (so 'yes' does not
    match 'eyes', nor 'equivalent' match 'inequivalent'); within each clause a word
    is negated iff a negation word precedes it in that clause. So 'not fully
    equivalent' and 'different, not identical' read NOT equivalent, while 'they do
    not diverge; identical' reads equivalent. Any surviving difference signal wins
    (conservative for a gate); unparseable -> NOT equivalent."""
    text = " ".join(_text_leaves(response.inline)).lower()
    difference = False
    equivalence = False
    for clause in re.split(r"[;,.\n!?:]", text):
        negated = False
        saw_polarity = False
        for word in _WORD.findall(clause):
            if word in _NEGATION_WORDS:
                negated = True
            elif word in _DIFFERENT_WORDS:
                saw_polarity = True
                equivalence = equivalence or negated  # "do not diverge" -> equivalent
                difference = difference or not negated
            elif word in _SAME_WORDS:
                saw_polarity = True
                difference = difference or negated  # "not equivalent" -> not equivalent
                equivalence = equivalence or not negated
        # A BARE negation — a clause with a negation word but no polarity word
        # ("...equivalent, but no.") — is a late reversal: count it as a difference
        # signal so it can't silently pass as EQUIVALENT (the one un-conservative
        # parse). Conservative for a gate: difference wins.
        if negated and not saw_polarity:
            difference = True
    if difference:
        return False  # any difference signal wins
    return equivalence  # equivalent iff a same-signal was seen; unparseable -> False


def _text_leaves(value: JSONValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _text_leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _text_leaves(item)]
    return []
