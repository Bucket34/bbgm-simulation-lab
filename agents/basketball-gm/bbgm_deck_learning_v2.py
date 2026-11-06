#!/usr/bin/env python3
"""BBGM Deck Learning Lab V2: learn a reusable GM policy in a separate toy simulator.

No actual Basketball GM engine is used. This script generates fictional team rosters
and free agents, trains a small *Python* value estimator using counterfactual
season outcomes from TRAINING scenarios, and tests it on entirely separate
UNSEEN scenarios against an overall-rating baseline and hindsight optimum.

- No external Python packages; no network access or Ollama needed.
- No changes to BBGMDeckAgentLab/agent_memory.json or existing V1 scripts.
- Model weights for Qwen3 are not changed; this is learned *decision policy* logic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import urllib.request
import urllib.error
from pathlib import Path

POSITIONS = ('PG', 'SG', 'SF', 'PF', 'C')
# These weights define the TOY simulator's hidden basketball environment.
# The learning estimator below receives player information, NOT these weights.
ROLE_WEIGHTS = {
    'PG': {'ovr': .60, 'pss': .26, 'tp': .08, 'diq': .06},
    'SG': {'ovr': .61, 'tp': .27, 'diq': .08, 'pss': .04},
    'SF': {'ovr': .62, 'tp': .14, 'diq': .17, 'reb': .07},
    'PF': {'ovr': .61, 'reb': .21, 'diq': .14, 'tp': .04},
    'C':  {'ovr': .59, 'reb': .27, 'diq': .14},
}
STATS = ('ovr', 'tp', 'pss', 'diq', 'reb')


def clamp(x, low, high):
    return max(low, min(high, x))


def player(rng, pos, pid, offer=False):
    base = rng.randint(55, 81) if offer else rng.randint(51, 73)
    specialty = rng.choice(('tp', 'pss', 'diq', 'reb', 'balanced'))
    record = {'id': pid, 'pos': pos, 'ovr': base,
              'cost': rng.randint(5, 12) if offer else 0,
              'age': rng.randint(20, 34)}
    for field in STATS[1:]:
        boost = 13 if field == specialty else (3 if specialty == 'balanced' else 0)
        record[field] = clamp(base + rng.randint(-19, 13) + boost, 10, 98)
    return record


def make_case(seed, number):
    # Separate, reproducible RNG per scenario; no overlap between train and test.
    rng = random.Random(seed + number * 100003)
    roster = [player(rng, p, f'S{i}') for i, p in enumerate(POSITIONS)]
    roster.extend(player(rng, rng.choice(POSITIONS), f'B{i}') for i in range(2))
    positions = rng.sample(POSITIONS, 4)
    offers = [player(rng, p, 'ABCD'[i], offer=True) for i, p in enumerate(positions)]
    budget = rng.randint(8, 12)
    valid = [p for p in offers if p['cost'] <= budget]
    # Guarantee at least two legal options; budget is a real constraint.
    if len(valid) < 2:
        budget = sorted(p['cost'] for p in offers)[1]
    return {'roster': roster, 'offers': offers, 'budget': budget,
            'id': number, 'seed': seed}


def role_score(p):
    return sum(p[k] * coef for k, coef in ROLE_WEIGHTS[p['pos']].items())


def team_strength(roster):
    # All five positions must be filled. Better positional fit matters more than
    # simply collecting the five highest overall-rated players.
    starters = [max(role_score(p) for p in roster if p['pos'] == pos) for pos in POSITIONS]
    # Mild general depth contribution from the best seven rated players.
    top = sorted((p['ovr'] for p in roster), reverse=True)[:7]
    depth = (sum(top) / len(top) - 60.0) * 0.06
    return sum(starters) / 5 + depth


def expected_wins(roster):
    probability = clamp(0.50 + (team_strength(roster) - 65.0) * 0.026, .12, .88)
    return 90.0 * probability


def legal_offers(case):
    return [p for p in case['offers'] if p['cost'] <= case['budget']]


def oracle_results(case):
    base = expected_wins(case['roster'])
    return {p['id']: expected_wins(case['roster'] + [p]) - base
            for p in legal_offers(case)}


def baseline_choice(case):
    return max(legal_offers(case), key=lambda p: (p['ovr'], -p['cost']))


# The policy sees only player/roster features, not the simulator's hidden
# ROLE_WEIGHTS or its expected-wins formula.
FEATURE_NAMES = (
    'bias', 'positive_ovr_upgrade', 'negative_ovr_upgrade',
    'positive_tp_gap', 'positive_pss_gap', 'positive_diq_gap', 'positive_reb_gap',
    'negative_tp_gap', 'negative_pss_gap', 'negative_diq_gap', 'negative_reb_gap',
    'cost', 'age',
    'position_PG', 'position_SG', 'position_SF', 'position_PF', 'position_C',
    'upgrade_PG', 'upgrade_SG', 'upgrade_SF', 'upgrade_PF', 'upgrade_C',
)


def features(case, offer):
    peers = [p for p in case['roster'] if p['pos'] == offer['pos']]
    current = max(peers, key=lambda p: p['ovr'])
    differences = {k: offer[k] - current[k] for k in STATS}
    ovr_gap = differences['ovr'] / 25
    values = [1., max(0., ovr_gap), min(0., ovr_gap)]
    values += [max(0., differences[s] / 40) for s in STATS[1:]]
    values += [min(0., differences[s] / 40) for s in STATS[1:]]
    values += [offer['cost'] / 12, offer['age'] / 35]
    values += [float(offer['pos'] == pos) for pos in POSITIONS]
    values += [max(0., ovr_gap) * float(offer['pos'] == pos) for pos in POSITIONS]
    assert len(values) == len(FEATURE_NAMES)
    return values


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def solve_linear(matrix, vector):
    """Small regularized system; standard-library Gauss-Jordan elimination."""
    n = len(vector)
    mat = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        mat[col], mat[pivot] = mat[pivot], mat[col]
        if abs(mat[col][col]) < 1e-12:
            raise RuntimeError('Ill-conditioned policy fitting matrix')
        divisor = mat[col][col]
        for j in range(col, n+1):
            mat[col][j] /= divisor
        for i in range(n):
            if i == col:
                continue
            scale = mat[i][col]
            for j in range(col, n+1):
                mat[i][j] -= scale * mat[col][j]
    return [mat[i][n] for i in range(n)]


def fit_policy(cases, regularization=2.0):
    n = len(FEATURE_NAMES)
    xx = [[0. for _ in range(n)] for _ in range(n)]
    xy = [0. for _ in range(n)]
    examples = 0
    for case in cases:
        rewards = oracle_results(case)
        for offer in legal_offers(case):
            x = features(case, offer)
            y = rewards[offer['id']]
            for i in range(n):
                xy[i] += x[i] * y
                for j in range(i, n):
                    xx[i][j] += x[i] * x[j]
            examples += 1
    for i in range(n):
        for j in range(i):
            xx[i][j] = xx[j][i]
        xx[i][i] += regularization
    return solve_linear(xx, xy), examples


def learned_choice(case, weights):
    return max(legal_offers(case),
               key=lambda p: (dot(weights, features(case, p)), -p['cost']))


def eval_policy(cases, weights):
    totals = {'ovr_baseline': 0., 'learned_policy': 0., 'oracle': 0.}
    sum_regret = {'ovr_baseline': 0., 'learned_policy': 0.}
    correct = {'ovr_baseline': 0, 'learned_policy': 0}
    records = []
    for case in cases:
        scores = oracle_results(case)
        best = max(scores, key=lambda c: scores[c])
        baseline = baseline_choice(case)['id']
        learned = learned_choice(case, weights)['id']
        baseline_regret = scores[best] - scores[baseline]
        learned_regret = scores[best] - scores[learned]
        for key, cid in (('ovr_baseline', baseline), ('learned_policy', learned), ('oracle', best)):
            # Expected wins include common starting roster strength, so different
            # scenarios with different teams can be compared fairly.
            totals[key] += expected_wins(case['roster']) + scores[cid]
        sum_regret['ovr_baseline'] += baseline_regret
        sum_regret['learned_policy'] += learned_regret
        correct['ovr_baseline'] += baseline_regret < 1e-9
        correct['learned_policy'] += learned_regret < 1e-9
        records.append({'case': case['id'], 'budget': case['budget'],
                        'baseline': baseline, 'learned': learned, 'oracle': best,
                        'expected_wins': {k: round(expected_wins(case['roster']) + v, 4)
                                          for k, v in scores.items()},
                        'regret_baseline': round(baseline_regret, 4),
                        'regret_learned': round(learned_regret, 4)})
    return totals, sum_regret, correct, records




def query_llm(prompt, model, base):
    payload = {
        'model': model, 'prompt': prompt, 'stream': False, 'think': False,
        'format': 'json', 'keep_alive': '2m',
        'options': {'num_ctx': 8192, 'num_predict': 220, 'temperature': 0, 'seed': 42},
    }
    request = urllib.request.Request(
        base.rstrip('/') + '/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=240) as r:
        result = json.load(r)
    if result.get('done_reason') == 'length':
        raise RuntimeError('Qwen reply truncated; decision not accepted')
    reply = json.loads(result.get('response', '{}'))
    choice = str(reply.get('choice', '')).strip().upper()
    return choice, str(reply.get('reason', ''))[:240], round(
        result.get('eval_count', 0) * 1e9 / max(1, result.get('eval_duration', 1)), 2)


def llm_prompt(case, ranked=None):
    intro = (
        'You are managing a team in an imaginary basketball simulator. '
        'Select ONE affordable free agent A/B/C/D to maximize next season wins. '
        'Every player has a fixed position. Positions, skill fit, and existing '
        'roster holes matter as well as overall rating. '
        'Return ONLY JSON {"choice":"A", "reason":"brief explanation"}. '
        'Do not assume any unprovided information.\n'
        f'Budget: {case["budget"]}\n'
        f'Current roster: {json.dumps(case["roster"], separators=(",", ":"))}\n'
        f'Offers: {json.dumps(case["offers"], separators=(",", ":"))}\n'
    )
    if ranked is not None:
        intro += (
            'A separate Python policy fitted on OTHER simulation cases produced '
            'these ESTIMATED win improvements for affordable options (not verified '
            'outcomes for this case). Consider them as advisory evidence, and '
            'select the best player: '
            f'{json.dumps(ranked, separators=(",", ":"))}\n'
        )
    return intro


def evaluate_llm(cases, weights, count, model, base):
    print(f'\n=== Qwen3 advisory comparison on {count} held-out cases ===', flush=True)
    print('Conditions: LLM alone vs. LLM given separate learned policy estimates.')
    print('Qwen3 weights remain unchanged. All choices are evaluated with expected wins.\n', flush=True)
    rows = []
    try:
        for case in cases[:count]:
            rewards = oracle_results(case)
            valid = legal_offers(case)
            rank = {p['id']: round(dot(weights, features(case, p)), 2) for p in valid}
            ids = set(rank)
            print(f'Scenario {case["id"] + 1}, budget {case["budget"]}:', flush=True)
            row = {'case': case['id'], 'policy_estimates': rank}
            for key, advice in [('qwen_alone', None), ('qwen_with_policy', rank)]:
                choice, reason, speed = query_llm(llm_prompt(case, advice), model, base)
                if choice not in ids:
                    raise RuntimeError(f'Invalid/unaffordable {key} choice: {choice!r}')
                row[key] = {'choice': choice, 'expected_wins': round(
                    expected_wins(case['roster']) + rewards[choice], 2),
                    'reason': reason, 'speed_tok_s': speed}
                print(f'  {key:<17} {choice} | expected wins: {row[key]["expected_wins"]:.2f} '
                      f'| {speed} tok/s', flush=True)
            policy_id = learned_choice(case, weights)['id']
            oracle_id = max(rewards, key=rewards.get)
            row['policy_choice'] = policy_id
            row['oracle_choice'] = oracle_id
            print(f'  Python policy     {policy_id} | oracle: {oracle_id}\n', flush=True)
            rows.append(row)
    finally:
        try:
            unload_req = urllib.request.Request(
                base.rstrip('/') + '/api/generate',
                data=json.dumps({'model': model, 'stream': False, 'keep_alive': 0}).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(unload_req, timeout=30):
                pass
        except Exception:
            pass
    print('=== LLM comparison (higher is better) ===')
    for label in ('qwen_alone', 'qwen_with_policy'):
        print(f'{label}: {sum(r[label]["expected_wins"] for r in rows)/len(rows):.2f} '
              'expected wins per scenario')
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=int, default=400, help='Number of synthetic training situations (default 400)')
    ap.add_argument('--test', type=int, default=100, help='Number of held-out synthetic situations (default 100)')
    ap.add_argument('--train-seed', type=int, default=4201)
    ap.add_argument('--test-seed', type=int, default=9901)
    ap.add_argument('--output', type=Path, default=Path.home() / 'BBGMDeckAgentLab')
    ap.add_argument('--llm-cases', type=int, default=0, help='Optional: compare Qwen3 alone vs Qwen3 with the learned policy on N unseen cases (0..5)')
    ap.add_argument('--model', default='qwen3:8b')
    ap.add_argument('--base', default='http://127.0.0.1:11435')
    args = ap.parse_args()
    if args.train < 20 or args.test < 1 or args.train > 10000 or args.test > 5000:
        ap.error('Please choose 20..10000 training and 1..5000 testing situations.')
    if not (0 <= args.llm_cases <= min(args.test, 5)):
        ap.error('--llm-cases must be 0..5 and at most --test')
    if args.train_seed == args.test_seed:
        ap.error('Training and testing seeds must differ.')

    print('Basketball GM Deck Learning Lab V2 — toy simulation only')
    print('No Ollama required; this trains a SMALL Python decision policy, NOT Qwen3 weights.')
    print('No actual Basketball GM leagues or V1 memories will be changed.\n')
    train_cases = [make_case(args.train_seed, i) for i in range(args.train)]
    test_cases = [make_case(args.test_seed, i) for i in range(args.test)]
    weights, examples = fit_policy(train_cases)
    totals, regrets, correct, records = eval_policy(test_cases, weights)
    print(f'Training cases: {args.train}; labeled legal offers: {examples}')
    print(f'Held-out cases: {args.test} (never used for fitting)\n')
    print('                 Avg expected wins / 90   Optimal picks   Regret to oracle')
    for label, key in [('OVR baseline', 'ovr_baseline'),
                       ('Learned policy', 'learned_policy'), ('Hindsight best', 'oracle')]:
        acc = '-' if key == 'oracle' else f'{correct[key]}/{args.test}'
        regret = '-' if key == 'oracle' else f'{regrets[key]/args.test:.3f}'
        print(f'{label:<17} {totals[key]/args.test:>10.2f}               {acc:>9}   {regret:>15}')
    improvement = (totals['learned_policy'] - totals['ovr_baseline']) / args.test
    print(f'\nLearned policy advantage vs OVR: {improvement:+.3f} expected wins per season.')
    print('These are EXPECTED values from the toy simulator, not real BBGM game results.')
    print('Counterfactual outcomes from all legal choices were available during training.')

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    report_path = args.output / f'learning-v2-{stamp}.json'
    policy_path = args.output / 'policy_v2.json'
    llm_rows = []
    if args.llm_cases:
        try:
            llm_rows = evaluate_llm(test_cases, weights, args.llm_cases, args.model, args.base)
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            print(f'LLM comparison failed: {exc}')
            print('The Python policy and offline evaluation completed; no prior files were changed.')
    output = {'kind': 'toy_policy_learning_v2', 'settings': vars(args) | {'output': str(args.output)},
              'feature_names': FEATURE_NAMES, 'weights': weights, 'training_offers': examples,
              'test_results': records, 'llm_comparison': llm_rows, 'total_expected_wins': totals,
              'optimal_picks': correct, 'total_expected_regret': regrets}
    report_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    policy_path.write_text(json.dumps({'feature_names': FEATURE_NAMES, 'weights': weights,
                                       'train_seed': args.train_seed, 'train_cases': args.train}, indent=2),
                           encoding='utf-8')
    print(f'\nSaved policy: {policy_path}\nSaved complete test report: {report_path}')


if __name__ == '__main__':
    main()
