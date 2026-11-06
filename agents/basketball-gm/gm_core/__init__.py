"""Reusable decision components for the synthetic BBGM GM lab.

The V5 basketball simulator and LLM client remain in the legacy experiment
scripts for now. This module does not call a model or alter saved league data.
"""
from .controller import DecisionResult, resolve_decision
from .policy import policy_pick
from .rationale import rationale_flags

__all__ = ['DecisionResult', 'resolve_decision', 'policy_pick', 'rationale_flags']
