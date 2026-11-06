"""Adapter for the frozen V5-compatible toy policy; not Qwen weight training."""
from __future__ import annotations

from . import simulator as sim


def policy_pick(episode, signings, year, params):
    """Return legal, positive-score policy pick and predicted offer scores.

    Preserves the V5.1 policy logic; evaluating a different environment
    will eventually require swapping this simulator dependency for an API.
    """
    legal = sim.available_offers(episode, signings, year)
    scores = sim.learned_estimates(episode, signings, year, params)
    best = max(legal, key=lambda p: (scores[p['id']], -p['cost']), default=None)
    if best is not None and scores[best['id']] > 0:
        return best['id'], scores
    return 'PASS', scores
