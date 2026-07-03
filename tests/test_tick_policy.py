"""BoundaryTickPolicy — automatic logical-tick sourcing for out-of-process runtimes
(§6, limitations gap #2)."""

from collections.abc import Sequence

from plumbline.core.seam import Seam
from plumbline.proxy.tick import BoundaryTickPolicy

_S = Seam.SENSOR_TO_CAPTION
_F = Seam.FUSE_TO_DECIDE


def _ticks(policy: BoundaryTickPolicy, seams: Sequence[Seam]) -> list[int]:
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
