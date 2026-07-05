"""Conformance checks for the `Adapter` contract (engineering spec §9.1).

`assert_conforms(adapter)` verifies an implementation satisfies the full, single
`Adapter` Protocol BEFORE it is driven — so a missing/wrong piece surfaces here with
a clear message, not much later as a counterfactual-replay divergence.

Two entry points:

  * `assert_conforms(adapter)` — raise `ConformanceError` naming the first failing
    piece (or nothing if it conforms).
  * `conformance_checks(adapter)` — a list of `(name, check)` pairs to drop straight
    into a parametrized pytest suite. No `pytest` import lives here; authors call it
    from their own tests, e.g.::

        import pytest
        from plumbline.adapters.conformance import conformance_checks

        @pytest.mark.parametrize(
            "name,check", conformance_checks(MyAdapter(...)), ids=lambda p: p
        )
        def test_adapter_conforms(name, check):
            check()
"""

from collections.abc import Callable
from typing import cast

from plumbline.adapters.base import ActionSchema, Adapter, BusTap
from plumbline.core.seam import Seam
from plumbline.core.trace import Payload

__all__ = ["ConformanceError", "assert_conforms", "conformance_checks"]

# The seven methods the unified Adapter Protocol requires (base.py). Kept explicit so
# a failure names the exact missing member rather than a bare "not an Adapter".
_REQUIRED_METHODS: tuple[str, ...] = (
    "configure_proxy",
    "bus_tap",
    "seam_of",
    "action_schema",
    "clock_hook",
    "reconstruct_caption_to_fuse",
    "reconstruct_decide_to_act",
)

# A benign probe payload/endpoint for exercising seam_of and parse without a runtime.
_PROBE_PAYLOAD = Payload(inline={})
_PROBE_ENDPOINT = ""


class ConformanceError(AssertionError):
    """An adapter does not satisfy the `Adapter` contract; the message names the
    specific missing or wrong piece. Subclasses AssertionError so it reads naturally
    when a check is invoked from a pytest suite."""


def _method(adapter: object, name: str) -> Callable[..., object]:
    """Fetch a bound method by name or raise a ConformanceError naming it. `name` is a
    variable, so this stays clear of the constant-getattr lint while keeping messages
    precise for the standalone checks (which may run on a not-yet-verified object)."""
    fn = getattr(adapter, name, None)
    if not callable(fn):
        raise ConformanceError(f"{_name(adapter)}.{name} is missing or not callable.")
    return cast(Callable[..., object], fn)


def _check_protocol(adapter: object) -> None:
    missing = [name for name in _REQUIRED_METHODS if not callable(getattr(adapter, name, None))]
    if missing:
        raise ConformanceError(
            f"{_name(adapter)} is missing required Adapter method(s): {', '.join(missing)}. "
            "The contract is seven methods (base.py); implement all of them."
        )
    if not isinstance(adapter, Adapter):
        raise ConformanceError(
            f"{_name(adapter)} does not satisfy the Adapter protocol despite having the "
            "expected method names — check signatures against plumbline.adapters.base.Adapter."
        )


def _check_seam_of(adapter: object) -> None:
    seam_of = _method(adapter, "seam_of")
    try:
        result = seam_of(_PROBE_PAYLOAD, _PROBE_ENDPOINT)
    except Exception as exc:  # noqa: BLE001 - surface any classifier error as a conformance failure
        raise ConformanceError(
            f"{_name(adapter)}.seam_of raised on a benign probe payload: {exc!r}. It must "
            "classify any captured call without raising."
        ) from exc
    if not isinstance(result, Seam):
        raise ConformanceError(
            f"{_name(adapter)}.seam_of returned {result!r}, not a plumbline.core.seam.Seam."
        )


def _check_action_schema(adapter: object) -> None:
    schema = _method(adapter, "action_schema")()
    if not isinstance(schema, ActionSchema):
        raise ConformanceError(
            f"{_name(adapter)}.action_schema() returned {schema!r}, which is not an "
            "ActionSchema (needs a `commands` attribute and a `parse` method)."
        )
    if not callable(getattr(schema, "parse", None)):
        raise ConformanceError(f"{_name(adapter)}.action_schema().parse is not callable.")
    try:
        parsed = schema.parse(_PROBE_PAYLOAD)
    except Exception as exc:  # noqa: BLE001 - a parser must tolerate an empty payload
        raise ConformanceError(
            f"{_name(adapter)}.action_schema().parse raised on an empty payload: {exc!r}. "
            "It must tolerate unknown/empty shapes and yield no actions."
        ) from exc
    if not isinstance(parsed, tuple):
        raise ConformanceError(
            f"{_name(adapter)}.action_schema().parse returned {parsed!r}, not a tuple of Actions."
        )


def _check_reconstruct_hooks(adapter: object) -> None:
    for name in ("reconstruct_caption_to_fuse", "reconstruct_decide_to_act"):
        if not callable(getattr(adapter, name, None)):
            raise ConformanceError(
                f"{_name(adapter)}.{name} is missing or not callable. Both reconstruct hooks "
                "are required so the four-seam producer can derive the two model-less seams."
            )


def _check_bus_tap(adapter: object) -> None:
    tap = _method(adapter, "bus_tap")()
    if tap is None:
        return  # a bus-less adapter (e.g. the generic template) is valid
    if not isinstance(tap, BusTap):
        raise ConformanceError(
            f"{_name(adapter)}.bus_tap() returned {tap!r}, which is neither None nor a BusTap "
            "(needs `key_expressions`, `subscribe`, and `close`)."
        )


def conformance_checks(adapter: object) -> list[tuple[str, Callable[[], None]]]:
    """Return named `(check_name, check)` pairs for the `Adapter` contract.

    Each `check` takes no arguments and raises `ConformanceError` on failure. Drop the
    list into a parametrized pytest test (see module docstring) or iterate it directly.
    """
    return [
        ("satisfies_protocol", lambda: _check_protocol(adapter)),
        ("seam_of_returns_seam", lambda: _check_seam_of(adapter)),
        ("action_schema_is_parseable", lambda: _check_action_schema(adapter)),
        ("reconstruct_hooks_present", lambda: _check_reconstruct_hooks(adapter)),
        ("bus_tap_is_none_or_valid", lambda: _check_bus_tap(adapter)),
    ]


def assert_conforms(adapter: object) -> None:
    """Raise `ConformanceError` if `adapter` does not satisfy the full `Adapter`
    contract; return None if it does. Runs every check in `conformance_checks`,
    protocol satisfaction first so a wholly wrong object fails with the clearest
    message."""
    for _, check in conformance_checks(adapter):
        check()


def _name(adapter: object) -> str:
    return type(adapter).__name__
