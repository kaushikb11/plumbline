"""The four seams of a language-bus runtime (engineering spec §3.1).

FROZEN (CLAUDE.md invariant 1): these four members are the contract. Naming the
seams this way is itself part of the contribution; do not add, rename, or
re-value a member to make a local problem easier.
"""

import enum


class Seam(enum.Enum):
    SENSOR_TO_CAPTION = "sensor_to_caption"  # raw frame/audio/state -> caption text
    CAPTION_TO_FUSE = "caption_to_fuse"  # captions + rules + RAG -> fused prompt
    FUSE_TO_DECIDE = "fuse_to_decide"  # fused prompt -> action plan
    DECIDE_TO_ACT = "decide_to_act"  # action plan -> HAL commands
