"""Baseline comparison — Experiment B (engineering spec §4, §7.6, §12).

The demo that sells the project: a config change (a swapped captioner) inverts the
robot's physical behavior, yet the observability a team would otherwise rely on
stays green. This harness runs, side by side on the same golden-vs-candidate run,
the observers Plumbline beats:

  - `latency_monitor` — OM1's Prometheus/Grafana latency stack. The loop is not
    slower, so it reports healthy.
  - `generic_tracer_monitor` — a Langfuse / OpenTelemetry-GenAI-style agent
    tracer. It sees well-formed LLM calls with plausible text outputs and no
    errors, so it flags nothing.
  - `plumbline_behavior_monitor` — scores the physical decision (the action
    sequence), so it goes red on the behavioral drift.

"Existing observability says fine, Plumbline says broken, the robot was in fact
broken" — as a checkable result, not an assertion (§12). Naming the baselines it
beats is what makes it a demo rather than a claim.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from plumbline.core.matcher import ExactMatcher, Matcher
from plumbline.core.seam import Seam
from plumbline.core.trace import JSONValue, Payload, SeamEvent
from plumbline.fidelity import structural_equivalence
from plumbline.proxy.otel import GEN_AI_OPERATION_NAME, seam_event_attributes
from plumbline.regression.golden import action_sequence

_MODEL_SEAMS = frozenset({Seam.SENSOR_TO_CAPTION, Seam.FUSE_TO_DECIDE})
_EXACT_MATCHER: Matcher = ExactMatcher()


@dataclass(frozen=True)
class MonitorVerdict:
    name: str
    healthy: bool  # True = green (looks fine), False = red (regression detected)
    detail: str


@dataclass(frozen=True)
class BaselineComparison:
    verdicts: tuple[MonitorVerdict, ...]

    @property
    def caught_by(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.verdicts if not v.healthy)

    @property
    def missed_by(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.verdicts if v.healthy)


def latency_monitor(
    golden: Sequence[SeamEvent],
    candidate: Sequence[SeamEvent],
    *,
    rel_tolerance: float = 0.25,
) -> MonitorVerdict:
    """OM1's latency stack: the loop got no slower, so it stays green."""
    golden_ms = _mean_model_latency(golden)
    candidate_ms = _mean_model_latency(candidate)
    if golden_ms == 0.0:
        # No baseline latency recorded: healthy only if the candidate is also 0,
        # not unconditionally (which would pass an arbitrarily slow candidate).
        healthy = candidate_ms == 0.0
    else:
        healthy = abs(candidate_ms - golden_ms) <= rel_tolerance * golden_ms
    return MonitorVerdict(
        "om1-latency", healthy, f"mean model latency {golden_ms:.1f}ms -> {candidate_ms:.1f}ms"
    )


def generic_tracer_monitor(candidate: Sequence[SeamEvent]) -> MonitorVerdict:
    """A generic OTel-GenAI / Langfuse-style tracer: sees well-formed LLM calls
    with plausible text and no errors, so it flags nothing — even when the caption
    is confidently wrong."""
    for event in _model_events(candidate):
        attrs = seam_event_attributes(event)  # the tracer reads the OTel-GenAI span
        is_genai_span = GEN_AI_OPERATION_NAME in attrs  # always set for a model seam
        text = _payload_text(event.response)
        inline = event.response.inline
        errored = isinstance(inline, dict) and "error" in inline
        # Flag only genuine problems (error body / empty output). Deliberately does
        # NOT require gen_ai.request.model — a real recording may omit it, and that
        # must not flip this baseline red and falsely invert the Experiment-B result.
        if errored or not text or not is_genai_span:
            return MonitorVerdict(
                "otel-genai-tracer", False, f"malformed or errored call at {event.seam.value}"
            )
    return MonitorVerdict("otel-genai-tracer", True, "all calls well-formed, outputs plausible")


def plumbline_behavior_monitor(
    golden: Sequence[SeamEvent],
    candidate: Sequence[SeamEvent],
    *,
    matcher: Matcher = _EXACT_MATCHER,
    drift_threshold: float = 0.0,
) -> MonitorVerdict:
    """Plumbline: scores the physical decision — the action sequence — so it
    catches the behavior inversion the text-level observers cannot (§7.5, §8.3)."""
    verdict = structural_equivalence(
        action_sequence(golden), action_sequence(candidate), matcher=matcher
    )
    healthy = verdict.distance <= drift_threshold
    return MonitorVerdict(
        "plumbline-behavior", healthy, f"behavioral drift {verdict.distance:.2f} ({verdict.reason})"
    )


def compare_against_baselines(
    golden: Sequence[SeamEvent],
    candidate: Sequence[SeamEvent],
    *,
    matcher: Matcher = _EXACT_MATCHER,
    drift_threshold: float = 0.0,
    latency_rel_tolerance: float = 0.25,
) -> BaselineComparison:
    """Run all three observers on the golden-vs-candidate run (§4 Experiment B)."""
    return BaselineComparison(
        verdicts=(
            latency_monitor(golden, candidate, rel_tolerance=latency_rel_tolerance),
            generic_tracer_monitor(candidate),
            plumbline_behavior_monitor(
                golden, candidate, matcher=matcher, drift_threshold=drift_threshold
            ),
        )
    )


def _model_events(events: Sequence[SeamEvent]) -> list[SeamEvent]:
    return [event for event in events if event.seam in _MODEL_SEAMS]


def _mean_model_latency(events: Sequence[SeamEvent]) -> float:
    model = _model_events(events)
    return sum(event.latency_ms for event in model) / len(model) if model else 0.0


def _text_leaves(value: JSONValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _text_leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _text_leaves(item)]
    return []


def _payload_text(payload: Payload) -> str:
    return " ".join(_text_leaves(payload.inline)).strip()
