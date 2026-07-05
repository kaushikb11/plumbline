"""CLI adapter resolution: built-ins, the entry-point group, and module:Class — each
conformance-checked so a wrong adapter fails at load with a clear message."""

import pytest
from plumbline.adapters.conformance import assert_conforms
from plumbline.cli import _adapter_from_entry_points, _build_adapter


def test_builtin_adapters_load_and_conform() -> None:
    for name in ("om1", "g1", "generic"):
        adapter = _build_adapter(name)
        assert_conforms(adapter)  # already checked inside, re-assert for clarity


def test_entry_point_group_resolves_registered_adapters() -> None:
    # The three built-ins are also registered under the 'plumbline.adapters' group,
    # which is the same mechanism a third-party runtime plugs into.
    adapter = _adapter_from_entry_points("generic")
    assert adapter is not None
    assert_conforms(adapter)
    assert _adapter_from_entry_points("does-not-exist") is None


def test_module_class_path_loads_a_third_party_adapter() -> None:
    adapter = _build_adapter("plumbline.adapters.generic:GenericAgentAdapter")
    assert_conforms(adapter)


def test_unknown_adapter_gives_an_actionable_error() -> None:
    with pytest.raises(ValueError, match="unknown adapter 'nope'"):
        _build_adapter("nope")


def test_non_conforming_class_is_rejected_at_load() -> None:
    # A class that isn't an adapter (built via module:Class) must fail conformance,
    # not silently load and diverge later.
    with pytest.raises(ValueError, match="does not satisfy the adapter contract"):
        _build_adapter("builtins:dict")
