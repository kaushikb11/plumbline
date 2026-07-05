"""Runtime adapters (engineering spec §9). OM1 is the flagship reference."""

from plumbline.adapters.action_matcher import ActionSchemaMatcher, recommended_behavior_matcher
from plumbline.adapters.base import (
    Action,
    ActionSchema,
    Adapter,
    BusSample,
    BusTap,
    ClockHook,
    ProxyConfig,
)
from plumbline.adapters.conformance import ConformanceError, assert_conforms, conformance_checks
from plumbline.adapters.g1 import G1ActionSchema, G1Adapter
from plumbline.adapters.generic import GenericActionSchema, GenericAgentAdapter
from plumbline.adapters.om1 import OM1ActionSchema, OM1Adapter

__all__ = [
    "Action",
    "ActionSchema",
    "ActionSchemaMatcher",
    "Adapter",
    "BusSample",
    "BusTap",
    "ClockHook",
    "ConformanceError",
    "assert_conforms",
    "conformance_checks",
    "G1ActionSchema",
    "G1Adapter",
    "GenericActionSchema",
    "GenericAgentAdapter",
    "OM1ActionSchema",
    "OM1Adapter",
    "ProxyConfig",
    "recommended_behavior_matcher",
]
