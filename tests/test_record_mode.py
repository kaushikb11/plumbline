"""Record-mode policy (the VCR ergonomic)."""

import pytest
from plumbline.record_mode import RecordMode, missing_trace_is_error, should_record


def test_parse_is_case_insensitive_and_validates() -> None:
    assert RecordMode.parse("NONE") is RecordMode.NONE
    assert RecordMode.parse(" once ") is RecordMode.ONCE
    assert RecordMode.parse(RecordMode.ALL) is RecordMode.ALL
    with pytest.raises(ValueError, match="unknown record mode"):
        RecordMode.parse("sometimes")


def test_should_record_matrix() -> None:
    # ALL always records; NONE never records; ONCE records only when absent.
    assert should_record(RecordMode.ALL, trace_exists=True) is True
    assert should_record(RecordMode.ALL, trace_exists=False) is True
    assert should_record(RecordMode.NONE, trace_exists=True) is False
    assert should_record(RecordMode.NONE, trace_exists=False) is False
    assert should_record(RecordMode.ONCE, trace_exists=False) is True
    assert should_record(RecordMode.ONCE, trace_exists=True) is False


def test_missing_trace_is_error_only_for_none() -> None:
    # NONE with no trace is the one combination a harness must refuse (no live call).
    assert missing_trace_is_error(RecordMode.NONE, trace_exists=False) is True
    assert missing_trace_is_error(RecordMode.NONE, trace_exists=True) is False
    assert missing_trace_is_error(RecordMode.ONCE, trace_exists=False) is False
    assert missing_trace_is_error(RecordMode.ALL, trace_exists=False) is False
