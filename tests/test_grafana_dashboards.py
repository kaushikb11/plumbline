"""Grafana dashboards parse and reference only fields the feeds actually emit (§11).

The last test is the honesty guard: a dashboard selector naming a field the feed
builders don't produce fails CI, so the dashboards can't drift into referencing
attributes the code never emits.
"""

import json
from pathlib import Path
from typing import Any

from plumbline.core.seam import Seam
from plumbline.core.trace import Episode, JSONValue, Payload, SeamEvent, canonicalize
from plumbline.observability.baselines import BaselineComparison, MonitorVerdict
from plumbline.observability.feed import baseline_feed, episode_telemetry, gate_feed
from plumbline.regression.gate import EpisodeDrift, FailurePolicy, GateResult

_GRAFANA_DIR = Path(__file__).resolve().parent.parent / "plumbline" / "observability" / "grafana"


def _all_keys(obj: JSONValue) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            keys |= _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


# Which feed(s) each dashboard binds to — its selectors are checked against ONLY
# these, so a telemetry panel cannot reference a gate-only field (or vice versa).
_DASHBOARD_FEEDS = {
    "plumbline-telemetry.json": ("telemetry",),
    "plumbline-regression.json": ("gate", "baseline"),
}


def _feed_keys() -> dict[str, set[str]]:
    request = Payload(inline={"m": 0})

    def event(seq: int, seam: Seam, response: JSONValue) -> SeamEvent:
        return SeamEvent(
            "ep",
            seq,
            seam,
            0,
            0.0,
            request,
            Payload(inline=response),
            "openai/gpt-4o",
            {},
            canonicalize(request).digest,
            1.0,
        )

    episode = Episode(
        "ep",
        (
            event(
                0, Seam.SENSOR_TO_CAPTION, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            ),
            event(1, Seam.FUSE_TO_DECIDE, {"choices": [{"message": {"content": "x"}}]}),
        ),
        {},
    )
    gate_result = GateResult(
        passed=False,
        threshold=0.1,
        policy=FailurePolicy.ANY,
        per_episode=(
            EpisodeDrift(
                "e1",
                drift=0.5,
                diverged=True,
                divergence_seam=Seam.CAPTION_TO_FUSE,
                divergence_distance=0.3,
            ),
        ),
    )
    comparison = BaselineComparison(verdicts=(MonitorVerdict("plumbline-behavior", False, "d"),))
    return {
        "telemetry": _all_keys(episode_telemetry(episode)),
        "gate": _all_keys(gate_feed(gate_result)),
        "baseline": _all_keys(baseline_feed(comparison)),
    }


def _selectors(obj: JSONValue) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("selector", "root_selector") and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_selectors(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_selectors(item))
    return found


def _dashboards() -> list[Path]:
    return sorted(_GRAFANA_DIR.glob("*.json"))


def test_dashboards_present() -> None:
    assert _dashboards()


def test_dashboards_parse_and_have_structure() -> None:
    for path in _dashboards():
        dashboard: Any = json.loads(path.read_text(encoding="utf-8"))
        assert dashboard["title"]
        assert isinstance(dashboard["panels"], list) and dashboard["panels"]
        var_types = {var["type"] for var in dashboard["templating"]["list"]}
        assert "datasource" in var_types  # templated datasource, not a hardcoded UID


def test_dashboards_reference_only_real_feed_fields() -> None:
    feed_keys = _feed_keys()
    for path in _dashboards():
        assert path.name in _DASHBOARD_FEEDS, f"{path.name}: add it to _DASHBOARD_FEEDS"
        allowed: set[str] = set()
        for feed_name in _DASHBOARD_FEEDS[path.name]:
            allowed |= feed_keys[feed_name]
        dashboard: JSONValue = json.loads(path.read_text(encoding="utf-8"))
        for selector in _selectors(dashboard):
            # Validate EVERY path segment against ONLY the feed(s) this dashboard binds
            # to — a wrong parent path or a cross-feed field name fails CI.
            for segment in selector.split("."):
                assert segment in allowed, (
                    f"{path.name}: selector {selector!r} references {segment!r}, not a feed field"
                )
