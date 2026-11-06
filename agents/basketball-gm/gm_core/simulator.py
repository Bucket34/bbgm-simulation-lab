"""Frozen-V5-compatible SYNTHETIC multi-season basketball environment.

These data structures, scoring formulas, salary rules, and deterministic
transitions are copied verbatim from bbgm_deck_multiseason_v5.py (V5).
The old V5 script is retained as an independent regression oracle.

This is NOT the actual ZenGM/Basketball GM engine. This module cannot
read/write BBGM league files and does not contact Ollama or execute AI output.

To integrate an external simulator later, define an adapter contract and
validate it separately; do not interpret these toy stats as Basketball GM data.
"""
from __future__ import annotations

import random

POSITIONS = ("PG", "SG", "SF", "PF", "C")


ATTRS = ("ovr", "tp", "pss", "diq", "reb")


ROLE_W = {
    "PG": {"ovr": .60, "pss": .26, "tp": .08, "diq": .06},
    "SG": {"ovr": .61, "tp": .27, "diq": .08, "pss": .04},
    "SF": {"ovr": .62, "tp": .14, "diq": .17, "reb": .07},
    "PF": {"ovr": .61, "reb": .21, "diq": .14, "tp": .04},
    "C": {"ovr": .59, "reb": .27, "diq": .14},
}


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def make_player(rng, position, pid, born_year, offer=False):
    rating = rng.randint(57, 81) if offer else rng.randint(54, 76)
    emphasis = rng.choice(("tp", "pss", "diq", "reb", "balanced"))
    age = rng.randint(20, 34) if offer else rng.randint(23, 33)
    p = {"id": pid, "pos": position, "age": age,
         "born_year": born_year, "cost": rng.randint(5, 14) if offer else 0,
         "length": rng.choice((2, 3)) if offer else 99,
         "ends": born_year + (rng.choice((2, 3)) if offer else 99),
         "ovr": rating}
    # Ensure declared contract length agrees with expiry.
    p["ends"] = born_year + p["length"]
    for stat in ATTRS[1:]:
        boost = 11 if stat == emphasis else (3 if emphasis == "balanced" else 0)
        p[stat] = clip(rating + rng.randint(-18, 13) + boost, 15, 98)
    return p


def make_episode(seed, idx, years):
    rng = random.Random(seed + 100003 * (idx + 1))
    initial = [make_player(rng, pos, f"Starter-{pos}", 0) for pos in POSITIONS]
    initial += [make_player(rng, rng.choice(POSITIONS), f"Bench-{i}", 0)
                for i in range(2)]
    offers_by_year = []
    for year in range(years):
        pos = rng.sample(POSITIONS, 4)
        offers_by_year.append([make_player(rng, pos[i], f"Y{year}-{'ABCD'[i]}",
                                                  year, offer=True) for i in range(4)])
    return {"id": idx, "initial": initial,
            "offers_by_year": offers_by_year,
            "cap": rng.randint(14, 22), "years": years}


def growth_step(age):
    # Entirely invented aging/development function for this synthetic arena.
    if age <= 22: return 2.8
    if age <= 25: return 1.7
    if age <= 28: return 0.3
    if age <= 31: return -1.0
    return -2.8


def evolved(p, year):
    elapsed = year - p["born_year"]
    assert elapsed >= 0
    delta = sum(growth_step(p["age"] + i) for i in range(elapsed))
    return {k: clip(p[k] + delta, 5, 99) for k in ATTRS}


def strength(team, year):
    if not team:
        return 0.0
    by_position = []
    for pos in POSITIONS:
        players = [p for p in team if p["pos"] == pos and p["ends"] > year]
        assert players, "initial team always provides all five positions"
        by_position.append(max(sum(evolved(p, year)[attr] * wt
                                   for attr, wt in ROLE_W[pos].items())
                               for p in players))
    overall = sorted((evolved(p, year)["ovr"] for p in team
                      if p["ends"] > year), reverse=True)[:7]
    return sum(by_position) / 5 + (sum(overall) / len(overall) - 60) * .06


def expected_wins(team, year):
    return 90 * clip(.50 + (strength(team, year) - 65) * .026, .12, .88)


def available_offers(episode, signings, year):
    active = [p for p in signings if p["ends"] > year]
    available = episode["cap"] - sum(p["cost"] for p in active)
    return [p for p in episode["offers_by_year"][year] if p["cost"] <= available]


def candidate_gains(episode, signings, year, offer):
    values = []
    for t in range(year, min(episode["years"], offer["ends"])):
        # No future offers are visible; only current signed players are projected.
        active = [p for p in signings if p["ends"] > t]
        present = episode["initial"] + active
        values.append(expected_wins(present + [offer], t) - expected_wins(present, t))
    return values


def choose_greedy(episode, signings, year):
    offers = available_offers(episode, signings, year)
    scores = {p["id"]: candidate_gains(episode, signings, year, p)[0]
              for p in offers}
    best = max(offers, key=lambda p: (scores[p["id"]], -p["cost"]), default=None)
    return best if best and scores[best["id"]] > 0.02 else None


def learned_estimates(episode, signings, year, params):
    discount, penalty, threshold = params
    offers = available_offers(episode, signings, year)
    scores = {}
    for p in offers:
        gains = candidate_gains(episode, signings, year, p)
        # Reward projected contributions over the contract; penalty captures
        # salary-cap opportunity cost from longer/more expensive contracts.
        future_value = sum((discount ** t) * delta for t, delta in enumerate(gains))
        cap_cost = p["cost"] * len(gains) / episode["cap"]
        scores[p["id"]] = future_value - penalty * cap_cost - threshold
    return scores


def choose_learned(episode, signings, year, params):
    offers = available_offers(episode, signings, year)
    scores = learned_estimates(episode, signings, year, params)
    choice = max(offers, key=lambda p: (scores[p["id"]], -p["cost"]), default=None)
    return choice if choice and scores[choice["id"]] > 0 else None


def run_episode(episode, choose, verbose=False):
    signings = []
    history = []
    total = 0.0
    for year in range(episode["years"]):
        active = [p for p in signings if p["ends"] > year]
        legal = available_offers(episode, signings, year)
        chosen = choose(episode, signings, year)
        if chosen is not None and chosen["id"] not in {p["id"] for p in legal}:
            raise ValueError("Illegal/over-budget policy choice")
        if chosen is not None:
            signings.append(chosen)
        roster = episode["initial"] + [p for p in signings if p["ends"] > year]
        wins = expected_wins(roster, year)
        total += wins
        history.append({"season": year + 1, "choice": chosen["id"] if chosen else "PASS",
                        "expected_wins": round(wins, 3),
                        "cap_remaining": episode["cap"] - sum(p["cost"] for p in signings
                                                                   if p["ends"] > year),
                        "legal_options": [p["id"] for p in legal]})
    return total, history


