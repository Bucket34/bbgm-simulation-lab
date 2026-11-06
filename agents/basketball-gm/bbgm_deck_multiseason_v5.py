#!/usr/bin/env python3
"""BBGM Deck Multi-Season Learning Lab V5 (independent synthetic sandbox).

Simulates a fictional 3-season roster-management process with yearly free-agent
choices, contract durations, salary limits, player aging and development.
A small Python *decision policy* is tuned on training episodes, then compared
with a greedy current-season baseline on entirely held-out episodes.

This is NOT the Basketball GM engine. It never touches BBGM saves, earlier V1-V4
lab files, or Qwen model weights. No packages needed beyond Python 3 stdlib.

Optional --llm-episodes N calls the existing local Ollama API for a separate,
limited comparison against the learned policy. The model can only recommend
player IDs. Python checks affordability and handles all state transitions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

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


def train(episodes):
    # Select the strategy weights ONLY on training episodes (no test exposure).
    search = list(itertools.product(
        (0.0, .35, .7, 1.0, 1.3),
        (0.0, .25, .75, 1.5, 3.0),
        (0.0, .25, .75, 1.5)))
    best = None
    for params in search:
        performance = sum(run_episode(ep, lambda e, s, y: choose_learned(e, s, y, params))[0]
                          for ep in episodes) / len(episodes)
        if best is None or performance > best[0] + 1e-10:
            best = (performance, params)
    return best, len(search)


def evaluate(episodes, params):
    greedy_tot = 0.0
    learned_tot = 0.0
    advantage = []
    details = []
    for ep in episodes:
        g, gh = run_episode(ep, choose_greedy)
        l, lh = run_episode(ep, lambda e, s, y: choose_learned(e, s, y, params))
        greedy_tot += g
        learned_tot += l
        advantage.append(l - g)
        details.append({"id": ep["id"], "salary_cap": ep["cap"],
                        "greedy_total_wins": round(g, 3), "learned_total_wins": round(l, 3),
                        "greedy_seasons": gh, "learned_seasons": lh})
    n = len(episodes)
    return {"greedy_expected_wins_per_season": greedy_tot/n/episodes[0]["years"],
            "learned_expected_wins_per_season": learned_tot/n/episodes[0]["years"],
            "delta_expected_wins_per_season": (learned_tot-greedy_tot)/n/episodes[0]["years"],
            "learned_better_episodes": sum(d > 1e-7 for d in advantage),
            "learned_worse_episodes": sum(d < -1e-7 for d in advantage),
            "tied_episodes": sum(abs(d) <= 1e-7 for d in advantage),
            "episodes": details}


def api(base, path, payload=None, timeout=240):
    raw = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base.rstrip('/') + path, data=raw,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_gpu(base, model):
    version = api(base, '/api/version', timeout=10).get('version')
    api(base, '/api/generate', {"model": model, "stream": False,
            "keep_alive": "5m", "options": {"num_ctx": 8192}})
    models = api(base, '/api/ps', timeout=10).get('models', [])
    loaded = next((m for m in models if m.get('name') == model), None)
    if not loaded or not loaded.get('size') or loaded['size'] != loaded['size_vram']:
        raise RuntimeError('Full Vulkan GPU offload not confirmed; skipping LLM experiment')
    print(f'Ollama {version}, GPU allocation: {loaded["size_vram"]:,}/{loaded["size"]:,}', flush=True)


def llm_choice(episode, signings, history, year, model, base, advice=None):
    legal = available_offers(episode, signings, year)
    legal_ids = [p['id'] for p in legal]
    cap_left = episode['cap'] - sum(p['cost'] for p in signings if p['ends'] > year)
    active_roster = episode['initial'] + [p for p in signings if p['ends'] > year]
    prompt_obj = {
        "role": "Fictional basketball general manager; choose ONE legal free-agent signing "
                "or PASS to maximize expected wins across all remaining seasons. "
                "Contracts block cap space until expiry. Young players develop; older players decline. "
                "Do not assume future free agents are known. Reply with ONLY JSON "
                '{"choice":"Y0-A" or "PASS", "reason":"short explanation"}.',
        "season": year+1, "seasons_remaining": episode['years'] - year,
        "salary_cap": episode['cap'], "available_cap": cap_left,
        "legal_candidate_ids": legal_ids + ['PASS'],
        "signed_roster": active_roster,
        "legal_offers": legal,
        "previous_seasons": history,
    }
    if advice is not None:
        prompt_obj['python_policy'] = {"description": "Estimated long-term value, not confirmed outcomes. "
                                          "May recommend PASS. You may disagree.",
                                        "scores": {k: round(v, 3) for k, v in advice.items()}}
    response = api(base, '/api/generate', {
        "model": model, "prompt": json.dumps(prompt_obj, separators=(',', ':')),
        "stream": False, "think": False, "format": "json", "keep_alive": "5m",
        "options": {"num_ctx": 8192, "num_predict": 220, "temperature": 0, "seed": 42},
    })
    result = {"choice": "PASS", "reason": "", "invalid": False,
              "finish": response.get('done_reason')}
    try:
        if response.get('done_reason') == 'length':
            raise ValueError('Truncated output')
        data = json.loads(response.get('response', ''))
        ident = str(data.get('choice', '')).strip().upper()
        if ident == 'PASS':
            result['choice'] = ident
        elif ident in legal_ids:
            result['choice'] = ident
        else:
            raise ValueError(f'Illegal choice: {ident}')
        result['reason'] = str(data.get('reason', ''))[:220]
    except (ValueError, TypeError, AttributeError) as e:
        result['invalid'] = True
        result['reason'] = f'Rejected decision, default PASS: {e}'
    duration = response.get('eval_duration') or 0
    result['speed'] = round(response.get('eval_count', 0) * 1e9 / duration, 2) if duration else None
    return result


def run_llm_episode(episode, params, model, base, advisory):
    signings = []
    history = []
    llm_records = []
    total = 0.0
    for year in range(episode['years']):
        advice = learned_estimates(episode, signings, year, params) if advisory else None
        response = llm_choice(episode, signings, history, year, model, base, advice)
        chosen = next((p for p in available_offers(episode, signings, year)
                       if p['id'] == response['choice']), None)
        if chosen:
            signings.append(chosen)
        roster = episode['initial'] + [p for p in signings if p['ends'] > year]
        wins = expected_wins(roster, year)
        total += wins
        record = {"season": year+1, "choice": response['choice'],
                  "expected_wins": round(wins, 3), "invalid": response['invalid'],
                  "reason": response['reason'], "speed": response['speed']}
        llm_records.append(record)
        history.append({"season": year+1, "choice": response['choice'],
                        "expected_wins": round(wins, 3)})
    return total, llm_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=37001)
    p.add_argument('--train', type=int, default=120)
    p.add_argument('--test', type=int, default=100)
    p.add_argument('--seasons', type=int, default=3)
    p.add_argument('--llm-episodes', type=int, default=0,
                   help='Optional held-out episodes to compare Qwen alone vs Qwen with policy advice')
    p.add_argument('--model', default='qwen3:8b')
    p.add_argument('--base', default='http://127.0.0.1:11435')
    p.add_argument('--home', type=Path, default=Path.home() / 'BBGMDeckAgentLab')
    args = p.parse_args()
    if not 30 <= args.train <= 1000 or not 10 <= args.test <= 500:
        p.error('train must be 30-1000; test 10-500')
    if not 2 <= args.seasons <= 5 or not 0 <= args.llm_episodes <= min(args.test, 5):
        p.error('seasons 2-5; llm-episodes 0-5 and at most test count')
    train_seed = args.seed
    test_seed = args.seed + 100000003
    training = [make_episode(train_seed, i, args.seasons) for i in range(args.train)]
    test = [make_episode(test_seed, i, args.seasons) for i in range(args.test)]
    print('BBGM Deck V5 — MULTI-SEASON SYNTHETIC GM LAB', flush=True)
    print('Toy simulator, NOT actual BBGM. Does not change prior policies, memories or league files.')
    print(f'Training episodes: {args.train}; held-out episodes: {args.test}; '
          f'seasons per episode: {args.seasons}')
    print('Features: aging/development, salary cap, multi-year signings, seasonal offers.')
    print('Future offers remain hidden from decision-makers.\n', flush=True)
    winner, candidates = train(training)
    train_avg, params = winner
    print(f'Trained small policy over {candidates} parameter combinations (training only).')
    print(f'Learned: future_discount={params[0]}, cap_penalty={params[1]}, '
          f'signing_threshold={params[2]}')
    print(f'Training expected wins per episode: {train_avg:.2f}')
    results = evaluate(test, params)
    print('\n=== HELD-OUT RESULTS ===')
    print(f'Greedy current-season GM: {results["greedy_expected_wins_per_season"]:.2f} '
          'expected wins/season')
    print(f'Learned long-horizon GM: {results["learned_expected_wins_per_season"]:.2f} '
          'expected wins/season')
    print(f'Policy improvement: {results["delta_expected_wins_per_season"]:+.3f} '
          'expected wins/season')
    print(f'Learned better / worse / tied episodes: '
          f'{results["learned_better_episodes"]} / '
          f'{results["learned_worse_episodes"]} / {results["tied_episodes"]}')
    report = {"experiment": "BBGM Deck V5 synthetic multi-season",
              "settings": {"seed": args.seed, "training": args.train,
                           "test": args.test, "seasons": args.seasons},
              "training_best_mean_episode_wins": train_avg,
              "learned_params": {"future_discount": params[0], "cap_penalty": params[1],
                                 "signing_threshold": params[2]},
              "held_out": results}
    if args.llm_episodes:
        print(f'\n=== QWEN3 HELD-OUT TRIAL ({args.llm_episodes} episodes) ===', flush=True)
        try:
            verify_gpu(args.base, args.model)
            llm_rows = []
            for i, episode in enumerate(test[:args.llm_episodes], 1):
                alone, alone_log = run_llm_episode(episode, params, args.model, args.base, False)
                advised, advised_log = run_llm_episode(episode, params, args.model, args.base, True)
                greedy, _ = run_episode(episode, choose_greedy)
                learned, _ = run_episode(episode, lambda e, s, y: choose_learned(e, s, y, params))
                print(f'Episode {i}: Qwen={alone/args.seasons:.2f}, '
                      f'Qwen+policy={advised/args.seasons:.2f}, '
                      f'greedy={greedy/args.seasons:.2f}, '
                      f'learned={learned/args.seasons:.2f} expected wins/season', flush=True)
                llm_rows.append({"episode": episode["id"],
                                 "qwen_alone_total": alone, "qwen_advised_total": advised,
                                 "greedy_total": greedy, "learned_total": learned,
                                 "qwen_alone_history": alone_log,
                                 "qwen_advised_history": advised_log})
            report['llm_trials'] = llm_rows
        except Exception as exc:
            print(f'LLM experiment failed: {exc}', file=sys.stderr)
            report['llm_error'] = str(exc)
        finally:
            try:
                api(args.base, '/api/generate', {"model": args.model,
                    "stream": False, "keep_alive": 0}, timeout=45)
                print('Model unloaded.')
            except Exception:
                pass
    args.home.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = args.home / f'multiseason-v5-{args.seed}-{stamp}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nSaved report: {path}')
    print('Only this new report was written. Qwen weights and older lab files remain unchanged.')


if __name__ == '__main__':
    main()
