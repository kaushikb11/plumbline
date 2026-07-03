"""The bus-message type shared by the bus tap and the adapter contract.

`BusSample` is a *transport* concept (a message observed on the runtime's bus), so
it lives in `transport/`, not `adapters/`. This placement breaks a package cycle:
the Zenoh tap (`transport/zenoh_tap.py`) needs `BusSample`, while adapters need the
tap — if `BusSample` lived in `adapters/base.py` (as it once did), transport would
import upward from adapters. `adapters/base.py` re-exports it for compatibility.
"""

from dataclasses import dataclass

from plumbline.core.trace import JSONValue


@dataclass(frozen=True)
class BusSample:
    """One message observed on the runtime's bus (e.g. a Zenoh sample).

    `payload` is the decoded semantic view (adapter decoder / JSON / text
    fallback); `raw` carries the exact wire bytes so the recorder can store them
    content-addressed — the decoded view is for comparison, the bytes are the
    ground truth (additive field; None for producers without a byte-level wire).
    """

    key_expr: str
    payload: JSONValue
    wall_ts: float
    raw: bytes | None = None
