"""Bus transport (engineering spec §9.2): the language-bus tap types.

`BusSample` is dependency-free and re-exported here. The Zenoh tap lives in
`plumbline.transport.zenoh_tap` (needs the `zenoh` extra) and is intentionally not
imported at package load so a base install stays dependency-free.
"""

from plumbline.transport.bus import BusSample

__all__ = ["BusSample"]
