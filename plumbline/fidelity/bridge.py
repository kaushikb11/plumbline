"""Bridge recorded seams into the decision-fidelity metrics (§7).

Closes the "fidelity not wired to recorded seams" limitation: a recorded episode
holds ONE decision per tick, but §7 needs decision *distributions* (and the §7.2
noise floor σ). The design: an opt-in, post-record sampling pass re-executes each
recorded FUSE_TO_DECIDE request N more times against the SAME endpoint and stores
the responses in a SIBLING samples episode (`<episode>.samples`) —

- the original trace stays byte-immutable (provenance, faithful replay untouched);
- the samples come from the same recorded model/session, not an analysis-time
  stand-in, and are themselves recorded (replayable evals, §7.5 discipline);
- nothing rides the recording hot path, so the runtime is never perturbed and
  the tick policy never sees the off-path calls.

HUMAN REVIEW (§14.5/§14.6, CLAUDE.md short leash): the sampling design above and
`default_decision_label` (tool-call canonicalization as the decision binning) are
judgment calls to be reviewed with the rest of the §7 math, not just tested.
"""

import json
from collections.abc import Callable, Sequence

from plumbline.core.clock import VirtualClock
from plumbline.core.recorder import Recorder
from plumbline.core.seam import Seam
from plumbline.core.store import TraceStore
from plumbline.core.trace import JSONValue, Payload, SeamEvent, canonical_dumps
from plumbline.fidelity.decision import (
    Distribution,
    Divergence,
    histogram,
    self_divergence,
    total_variation,
)
from plumbline.fidelity.metrics import DecisionDrift

# Re-execute a recorded request against the same endpoint; returns the response.
PostFn = Callable[[Payload], Payload]
# Bin a recorded decision response into a decision-class label (§7.1, §14.6).
ResponseLabel = Callable[[Payload], str]

_SAMPLES_OF_KEY = "plumbline.samples_of"
_SAMPLE_INDEX_KEY = "plumbline.sample_index"


def samples_episode_id(episode_id: str) -> str:
    return f"{episode_id}.samples"


def default_decision_label(response: Payload) -> str:
    """Canonical label for a recorded Cortex response: the (name, arguments) list of
    its tool calls, else the message text, else the canonical inline content.

    Lossless by default like `canonical_label` (§14.6): distinct decisions never
    collapse; inject a coarser label to bin (e.g. tolerance-bucketed arguments).
    """
    inline = response.inline
    if isinstance(inline, dict):
        choices = inline.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                calls = message.get("tool_calls")
                if isinstance(calls, list) and calls:
                    labeled: list[JSONValue] = []
                    for call in calls:
                        if not isinstance(call, dict):
                            continue
                        function = call.get("function")
                        if not isinstance(function, dict):
                            continue
                        arguments = function.get("arguments")
                        try:
                            parsed = (
                                json.loads(arguments) if isinstance(arguments, str) else arguments
                            )
                        except ValueError:
                            parsed = arguments
                        labeled.append({"name": function.get("name"), "arguments": parsed})
                    return canonical_dumps(labeled)
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
    return canonical_dumps(inline)


def sample_recorded_decisions(
    store: TraceStore,
    episode_id: str,
    post: PostFn,
    n: int,
    *,
    seam: Seam = Seam.FUSE_TO_DECIDE,
) -> str:
    """Re-execute each recorded `seam` request N more times against the same
    endpoint, recording the responses into the sibling samples episode. Returns the
    samples episode id. The original episode is not touched."""
    episode = store.load_episode(episode_id)
    recorder = Recorder(store, VirtualClock())
    sibling = samples_episode_id(episode_id)
    recorder.open_episode(sibling, {_SAMPLES_OF_KEY: episode_id, "plumbline.samples_n": n})
    seq = 0
    for event in episode.events:
        if event.seam is not seam:
            continue
        for index in range(n):
            response = post(event.request)
            recorder.record(
                SeamEvent(
                    episode_id=sibling,
                    seq=seq,
                    seam=seam,
                    logical_tick=event.logical_tick,
                    wall_ts=0.0,  # post-hoc sample; never drives replay (§3.2)
                    request=event.request,
                    response=response,
                    model_id=event.model_id,
                    params={_SAMPLES_OF_KEY: episode_id, _SAMPLE_INDEX_KEY: index},
                    request_digest=event.request_digest,
                    latency_ms=0.0,
                )
            )
            seq += 1
    recorder.close_episode(sibling)
    return sibling


def recorded_labels(
    store: TraceStore,
    episode_id: str,
    tick: int,
    *,
    seam: Seam = Seam.FUSE_TO_DECIDE,
    label_of: ResponseLabel = default_decision_label,
    include_original: bool = True,
) -> list[str]:
    """The decision-label sample at one tick: the sibling episode's N samples,
    plus (by default) the on-path recorded decision itself."""
    labels: list[str] = []
    if include_original:
        for event in store.load_episode(episode_id).events:
            if event.seam is seam and event.logical_tick == tick:
                labels.append(label_of(event.response))
    for event in store.load_episode(samples_episode_id(episode_id)).events:
        if event.seam is seam and event.logical_tick == tick:
            labels.append(label_of(event.response))
    return labels


def recorded_distribution(
    store: TraceStore,
    episode_id: str,
    tick: int,
    *,
    seam: Seam = Seam.FUSE_TO_DECIDE,
    label_of: ResponseLabel = default_decision_label,
) -> Distribution:
    """`D(recorded context at tick)` estimated from the recorded samples (§7.1)."""
    return histogram(recorded_labels(store, episode_id, tick, seam=seam, label_of=label_of))


def recorded_decision_drift(
    store: TraceStore,
    episode_id: str,
    tick: int,
    candidate_responses: Sequence[Payload],
    *,
    seam: Seam = Seam.FUSE_TO_DECIDE,
    label_of: ResponseLabel = default_decision_label,
    divergence: Divergence = total_variation,
    trials: int = 32,
    seed: int = 0,
) -> DecisionDrift:
    """Decision divergence of a candidate (e.g. a counterfactual's responses at this
    tick) from the RECORDED decision distribution, corrected by the recorded noise
    floor σ (§7.2): excess = max(0, div − σ). σ is the split-half self-divergence of
    the recorded sample, so the floor reflects the recorded decider's own noise."""
    golden_labels = recorded_labels(store, episode_id, tick, seam=seam, label_of=label_of)
    sigma = self_divergence(golden_labels, divergence=divergence, trials=trials, seed=seed)
    div = divergence(
        histogram(golden_labels),
        histogram([label_of(response) for response in candidate_responses]),
    )
    return DecisionDrift(divergence=div, sigma=sigma, excess=max(0.0, div - sigma))
