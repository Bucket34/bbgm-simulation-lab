"""Deterministic decision validation and policy overrides, no LLM dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping, Any


@dataclass(frozen=True)
class DecisionResult:
    """What the LLM proposed and what the controller will execute.

    ``estimated_gain`` is in learned-policy SCORE units, not simulated wins.
    The policy is not an oracle, and this value is not a confidence score.
    """

    proposal_id: str
    invalid: bool
    effective_proposal: str
    final_action: str
    estimated_gain: float
    override_reason: str


def resolve_decision(
    proposal: Mapping[str, Any],
    legal_ids: Collection[str],
    recommended: str,
    scores: Mapping[str, float],
    mode: str,
    margin: float,
) -> DecisionResult:
    """Choose an action without side effects or access to hidden outcomes.

    Accept only PASS or one of the legal offer IDs. In unguarded modes an
    invalid AI proposal becomes PASS. The guarded mode may instead select
    the legal Python-policy recommendation, or override when its estimated
    score exceeds Qwen's by at least ``margin``.
    """
    if mode not in ('alone', 'advised', 'guarded'):
        raise ValueError(f'Unknown decision mode: {mode}')
    if margin < 0:
        raise ValueError('Override margin must be nonnegative')
    choices = set(legal_ids)
    if recommended != 'PASS' and recommended not in choices:
        raise ValueError('Policy recommendation is not a legal offer')

    proposal_id = str(proposal.get('choice', 'PASS')).strip().upper()
    invalid = bool(proposal.get('invalid', False)) or (
        proposal_id != 'PASS' and proposal_id not in choices
    )
    effective_proposal = 'PASS' if invalid else proposal_id
    chosen = effective_proposal
    override_reason = ''
    estimated_gain = round(
        scores.get(recommended, 0.0) - scores.get(effective_proposal, 0.0), 3
    )

    if mode == 'guarded':
        if invalid:
            chosen = recommended
            override_reason = 'Invalid AI proposal; use legal policy recommendation'
        elif recommended != effective_proposal and estimated_gain >= margin:
            chosen = recommended
            override_reason = (
                f'Policy estimated gain {estimated_gain:+.3f} '
                f'>= threshold {margin:.3f}'
            )
    return DecisionResult(
        proposal_id, invalid, effective_proposal, chosen,
        estimated_gain, override_reason,
    )
