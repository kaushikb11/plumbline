"""Record modes (the VCR.py ergonomic) — how a record/replay harness decides, per
episode, whether to record a live run or replay a stored trace.

    NONE  replay only; a missing trace or an unrecorded live call is an ERROR.
          This is the CI default: a green run PROVES nothing hit a live model.
    ONCE  replay if the episode's trace exists, else record it live once.
    ALL   always (re-)record live, overwriting any existing trace.

The mode is pure policy; `should_record` maps (mode, trace_exists) to record-vs-replay,
and `missing_trace_is_error` flags the NONE-with-no-trace case a harness must reject.
"""

import enum


class RecordMode(enum.Enum):
    NONE = "none"
    ONCE = "once"
    ALL = "all"

    @classmethod
    def parse(cls, value: "str | RecordMode") -> "RecordMode":
        """Coerce a string (case-insensitive) or RecordMode into a RecordMode."""
        if isinstance(value, RecordMode):
            return value
        try:
            return cls(value.strip().lower())
        except ValueError:
            choices = ", ".join(m.value for m in cls)
            raise ValueError(f"unknown record mode {value!r}; choose one of: {choices}") from None


def should_record(mode: RecordMode, trace_exists: bool) -> bool:
    """True → record a live run; False → replay the stored trace.

    (NONE always replays; a NONE run with no trace is an error the caller surfaces
    via `missing_trace_is_error` — don't silently fall through to a live call.)"""
    if mode is RecordMode.ALL:
        return True
    if mode is RecordMode.ONCE:
        return not trace_exists
    return False  # NONE


def missing_trace_is_error(mode: RecordMode, trace_exists: bool) -> bool:
    """True when the harness must refuse to run: NONE mode with no trace to replay."""
    return mode is RecordMode.NONE and not trace_exists
