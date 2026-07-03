"""BoundaryTickPolicy / PerCallTickPolicy — automatic logical-tick sourcing for
out-of-process runtimes (§6, limitations gap #2)."""

from collections.abc import Sequence

from plumbline.core.seam import Seam
from plumbline.proxy.tick import BoundaryTickPolicy, PerCallTickPolicy, TickPolicy

_S = Seam.SENSOR_TO_CAPTION
_F = Seam.FUSE_TO_DECIDE
_D = Seam.DECIDE_TO_ACT


def _ticks(policy: TickPolicy, seams: Sequence[Seam]) -> list[int]:
    return [policy.next_tick(seam, None) for seam in seams]


def test_boundary_advances_once_per_cycle() -> None:
    assert _ticks(BoundaryTickPolicy(), [_S, _F, _S, _F]) == [0, 0, 1, 1]


def test_multiple_captions_share_a_tick() -> None:
    assert _ticks(BoundaryTickPolicy(), [_S, _S, _F]) == [0, 0, 0]


def test_first_event_is_tick_zero() -> None:
    assert BoundaryTickPolicy().next_tick(_S, None) == 0


def test_loop_with_no_boundary_collapses_to_one_tick() -> None:
    # Documented scoped behavior: no boundary seam -> one tick (configure boundary_seam
    # or send the header for such runtimes).
    assert _ticks(BoundaryTickPolicy(), [_F, _F, _F]) == [0, 0, 0]


def test_header_override_wins_and_resyncs() -> None:
    policy = BoundaryTickPolicy()
    assert policy.next_tick(_S, None) == 0
    assert policy.next_tick(_F, 5) == 5  # explicit override wins
    assert policy.next_tick(_S, None) == 6  # auto-advance continues from the override


# --- PerCallTickPolicy: pure decide loops (found against the real OM1 binary) ---


def test_per_call_each_boundary_call_is_its_own_tick() -> None:
    # The case BoundaryTickPolicy collapses: every seam IS the boundary seam.
    assert _ticks(PerCallTickPolicy(), [_F, _F, _F]) == [0, 1, 2]


def test_per_call_non_boundary_stays_in_current_cycle() -> None:
    assert _ticks(PerCallTickPolicy(), [_F, _D, _F, _D]) == [0, 0, 1, 1]


def test_per_call_pre_boundary_seam_lands_in_first_cycle() -> None:
    assert _ticks(PerCallTickPolicy(), [_D, _F, _F]) == [0, 0, 1]


def test_per_call_override_wins_and_resyncs() -> None:
    policy = PerCallTickPolicy()
    assert policy.next_tick(_F, None) == 0
    assert policy.next_tick(_F, 7) == 7  # explicit override wins
    assert policy.next_tick(_F, None) == 8  # auto-advance continues from the override
