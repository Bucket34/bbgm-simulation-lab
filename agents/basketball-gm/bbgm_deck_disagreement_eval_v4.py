#!/usr/bin/env python3
"""BBGM Deck Learning Lab V4: held-out evaluation + disagreement stress test.

This is a synthetic TOY simulator; it does not run Basketball GM.
- Reads the existing policy_v2.json; does NOT update the trained policy.
- Evaluates the policy against OVR on many new held-out synthetic situations.
- Selects a fixed subset for optional Qwen3 testing using ONLY policy/OVR
  disagreement (NOT oracle outcomes). This subset is intentionally enriched
  for disagreements, so its results do not estimate general population wins.
- Never changes Qwen3 weights, league files or the prior agent_memory.json.

Requires Python 3 standard library and bbgm_deck_learning_v2.py in the
same folder. Ollama is needed only when --run-llm is specified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

import bbgm_deck_learning_v2 as sim


def api(base, path, payload=None, timeout=240):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(
        base.rstrip('/') + path, data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def load_policy(path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    if obj.get('feature_names') != list(sim.FEATURE_NAMES):
        raise ValueError('Policy feature names do not match V2 simulator')
    w = obj.get('weights')
    if not isinstance(w, list) or len(w) != len(sim.FEATURE_NAMES):
        raise ValueError('Policy weights have incorrect shape')
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in w):
        raise ValueError('Policy weights must be finite numbers')
    return w, obj


def estimates(case, w):
    return {p['id']: sim.dot(w, sim.features(case, p))
            for p in sim.legal_offers(case)}


def evaluate_population(cases, w):
    totals = {'ovr': 0., 'policy': 0., 'oracle': 0.}
    optimal = {'ovr': 0, 'policy': 0}
    disagree = 0
    for case in cases:
        results = sim.oracle_results(case)
        ovr = sim.baseline_choice(case)['id']
        policy = sim.learned_choice(case, w)['id']
        best_val = max(results.values())
        base = sim.expected_wins(case['roster'])
        totals['ovr'] += base + results[ovr]
        totals['policy'] += base + results[policy]
        totals['oracle'] += base + best_val
        optimal['ovr'] += best_val - results[ovr] < 1e-8
        optimal['policy'] += best_val - results[policy] < 1e-8
        disagree += ovr != policy
    n = len(cases)
    return {
        'evaluated': n,
        'means': {k: v/n for k, v in totals.items()},
        'optimal_picks': optimal,
        'policy_vs_ovr_expected_wins': (totals['policy']-totals['ovr'])/n,
        'baseline_policy_disagreements': disagree,
    }


def pick_disagreements(cases, w, count, min_gap, selection_seed):
    # IMPORTANT: No oracle_results call in selection to prevent outcome leakage.
    eligible = []
    for case in cases:
        ovr = sim.baseline_choice(case)['id']
        policy = sim.learned_choice(case, w)['id']
        predicted = estimates(case, w)
        advantage = predicted[policy] - predicted[ovr]
        if policy != ovr and advantage >= min_gap:
            eligible.append((case, ovr, policy, advantage))
    if len(eligible) < count:
        raise ValueError(f'Only {len(eligible)} eligible disagreements; '
                         f'reduce --cases or --min-gap')
    chosen = random.Random(selection_seed).sample(eligible, count)
    return sorted(chosen, key=lambda entry: entry[0]['id']), len(eligible)


def verify_vulkan(base, model):
    version = api(base, '/api/version', timeout=10).get('version', '?')
    api(base, '/api/generate', {
        'model': model, 'stream': False, 'keep_alive': '5m',
        'options': {'num_ctx': 8192},
    })
    active = api(base, '/api/ps', timeout=10).get('models', [])
    loaded = next((x for x in active if x.get('name') == model), None)
    if loaded is None:
        raise RuntimeError(f'{model} was not reported as loaded')
    size = loaded.get('size', 0)
    vram = loaded.get('size_vram', 0)
    print(f'Ollama {version}; GPU allocation: {vram:,}/{size:,} bytes', flush=True)
    if not size or size != vram:
        raise RuntimeError('Full GPU offloading not confirmed; do not run CPU-only comparison')


def ask_qwen(case, model, base, advice=None):
    allowed = sim.legal_offers(case)
    legal_ids = [p['id'] for p in allowed]
    content = {
        'instruction': 'You are an AI general manager in a fictional basketball simulator. '
                       'Select exactly one affordable candidate to maximize next-season wins. '
                       'Consider positions, skill fit, roster needs, and player attributes. '
                       'Respond ONLY with JSON: {"choice":"A","reason":"brief explanation"}. '
                       'Choose only from legal_candidate_ids.',
        'legal_candidate_ids': legal_ids,
        'budget': case['budget'],
        'roster': case['roster'],
        'legal_offers': allowed,
    }
    if advice is not None:
        content['advice'] = {
            'description': 'Independent Python policy predictions, learned on OTHER synthetic cases. '
                           'Approximate improvements in expected wins, NOT proven outcomes. '
                           'You may accept or reject the advice.',
            'predicted_improvement_by_player': {
                k: round(v, 3) for k, v in advice.items()
            },
        }
    result = api(base, '/api/generate', {
        'model': model, 'prompt': json.dumps(content, separators=(',', ':')),
        'stream': False, 'think': False, 'format': 'json',
        'keep_alive': '5m',
        'options': {'num_ctx': 8192, 'num_predict': 220,
                    'temperature': 0, 'seed': 42},
    })
    output = {'invalid': False, 'reason': '', 'choice': None,
              'speed_tok_s': 0., 'finish_reason': result.get('done_reason')}
    duration = result.get('eval_duration', 0)
    if duration:
        output['speed_tok_s'] = round(result.get('eval_count', 0)*1e9/duration, 2)
    if result.get('done_reason') == 'length':
        output['invalid'] = True
        output['reason'] = 'Truncated model output'
        return output
    try:
        decoded = json.loads(result.get('response', ''))
        chosen = str(decoded.get('choice', '')).strip().upper()
        output['reason'] = str(decoded.get('reason', ''))[:200]
        if chosen in legal_ids:
            output['choice'] = chosen
        else:
            output['invalid'] = True
            output['reason'] = 'Invalid or unaffordable choice: '+repr(chosen)
    except (ValueError, TypeError, AttributeError):
        output['invalid'] = True
        output['reason'] = 'Response was not a valid JSON decision'
    return output


def run_llm(selected, w, model, base, margin):
    verify_vulkan(base, model)
    rows = []
    for idx, (case, ovr, policy, pred_gap) in enumerate(selected):
        observed = sim.oracle_results(case)
        base_wins = sim.expected_wins(case['roster'])
        pred = estimates(case, w)
        order = ('alone', 'advised') if idx % 2 == 0 else ('advised', 'alone')
        llm = {}
        for mode in order:
            llm[mode] = ask_qwen(case, model, base, pred if mode == 'advised' else None)
        alone = llm['alone']['choice']
        advised = llm['advised']['choice']
        def wins(chosen):
            return base_wins + observed[chosen] if chosen else base_wins
        predicted_gain = (pred[policy] - pred[alone]) if alone else None
        hybrid = policy if (alone is None or predicted_gain >= margin) else alone
        oracle_wins = base_wins + max(observed.values())
        choices = {'qwen_alone': alone, 'qwen_advised': advised,
                   'python_policy': policy, 'hybrid': hybrid, 'ovr_baseline': ovr}
        scores = {key: wins(choice) for key, choice in choices.items()}
        scores['oracle'] = oracle_wins
        row = {'scenario_id': case['id'], 'budget': case['budget'],
               'legal_candidates': [o['id'] for o in sim.legal_offers(case)],
               'predicted_policy_advantage_over_ovr': pred_gap,
               'choices': choices, 'scores': scores, 'model_replies': llm,
               'oracle': max(observed, key=observed.get)}
        rows.append(row)
        print(f"  Case {case['id']+1}: Qwen {alone or 'INVALID'}={scores['qwen_alone']:.2f} | "
              f"advised {advised or 'INVALID'}={scores['qwen_advised']:.2f} | "
              f"policy {policy}={scores['python_policy']:.2f} | "
              f"hybrid {hybrid}={scores['hybrid']:.2f} | "
              f"oracle {scores['oracle']:.2f}", flush=True)
    return rows


def summarize_llm(rows):
    keys = ('qwen_alone', 'qwen_advised', 'python_policy', 'hybrid', 'ovr_baseline', 'oracle')
    print('\n=== Disagreement stress test: EXPECTED WINS ===')
    print(f'{"Strategy":<17} {"Mean wins":>11} {"Optimal":>10} {"Regret":>10}')
    summary = {}
    for key in keys:
        mean = sum(r['scores'][key] for r in rows)/len(rows)
        optimal = sum(abs(r['scores']['oracle']-r['scores'][key]) < 1e-7 for r in rows)
        regret = sum(r['scores']['oracle']-r['scores'][key] for r in rows)/len(rows)
        summary[key] = {'mean_wins': mean, 'optimal_count': optimal, 'mean_regret': regret}
        print(f'{key:<17} {mean:>11.2f} {optimal:>7}/{len(rows):<2} {regret:>10.3f}')
    print('Enriched disagreement sample, NOT a representative estimate of overall manager skill.')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=26001)
    parser.add_argument('--pool', type=int, default=500)
    parser.add_argument('--cases', type=int, default=6)
    parser.add_argument('--min-gap', type=float, default=0.5,
                        help='Minimum predicted policy advantage over OVR for selected cases')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Hybrid predicted-win advantage needed to override Qwen choice')
    parser.add_argument('--run-llm', action='store_true',
                        help='Also compare Qwen alone vs advised on chosen disagreement cases')
    parser.add_argument('--home', type=Path, default=Path.home() / 'BBGMDeckAgentLab')
    parser.add_argument('--base', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='qwen3:8b')
    args = parser.parse_args()
    if not 100 <= args.pool <= 5000:
        parser.error('--pool must be 100..5000')
    if not 1 <= args.cases <= 20:
        parser.error('--cases must be 1..20')
    if not all(math.isfinite(x) and x >= 0 for x in (args.min_gap, args.margin)):
        parser.error('--min-gap and --margin must be finite and nonnegative')
    if args.seed in (4201, 9901, 16001):
        parser.error('Choose a new seed separate from earlier training and tests')
    weights, saved = load_policy(args.home / 'policy_v2.json')
    if saved.get('train_seed') == args.seed:
        parser.error('Test seed must differ from policy training seed')

    print('BBGM Deck V4 -- held-out toy-policy check + informative disagreement cases')
    print('Synthetic simulator ONLY. Does not modify the policy, saved history, BBGM leagues or Qwen3 weights.')
    cases = [sim.make_case(args.seed, idx) for idx in range(args.pool)]
    population = evaluate_population(cases, weights)
    print(f'\n=== Unseen sample: {args.pool} scenarios ===')
    for label, key in [('OVR baseline', 'ovr'), ('Python policy', 'policy'), ('Oracle', 'oracle')]:
        optimal = '-' if key == 'oracle' else f"{population['optimal_picks'][key]}/{args.pool}"
        print(f"  {label:<16} mean {population['means'][key]:.2f} expected wins, optimal {optimal}")
    print(f"  Policy improvement: {population['policy_vs_ovr_expected_wins']:+.3f} expected wins/scenario")
    print(f"  Baseline/policy disagreements: {population['baseline_policy_disagreements']}/{args.pool}")

    chosen, eligible = pick_disagreements(cases, weights, args.cases, args.min_gap, args.seed+9917)
    print(f'\n=== Selected {args.cases} disagreement stress-test cases ===')
    print(f'Selected randomly from {eligible} policy/OVR disagreements with predicted gap >= {args.min_gap:.2f}.')
    print('Selection uses NO true simulation outcomes, but this subset is deliberately non-representative.')
    for case, ovr, policy, advantage in chosen:
        print(f"  Scenario {case['id']+1}: OVR {ovr} vs policy {policy}; "
              f"predicted policy advantage {advantage:+.2f} wins")
    report = {
        'experiment': 'toy_bbgm_v4_disagreement',
        'settings': {'seed': args.seed, 'pool': args.pool, 'cases': args.cases,
                     'min_gap': args.min_gap, 'margin': args.margin, 'model': args.model},
        'offline_population': population,
        'stress_sample': [{'case': case, 'ovr_choice': ovr, 'policy_choice': policy,
                           'predicted_advantage': advantage}
                          for case, ovr, policy, advantage in chosen],
    }
    if args.run_llm:
        print('\n=== Qwen3 evaluation ===', flush=True)
        try:
            rows = run_llm(chosen, weights, args.model, args.base, args.margin)
            report['llm_scenarios'] = rows
            report['llm_summary'] = summarize_llm(rows)
        except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            print(f'LLM test failed: {exc}; offline results are still available.', file=sys.stderr)
        finally:
            try:
                api(args.base, '/api/generate', {'model': args.model, 'stream': False,
                      'keep_alive': 0}, timeout=45)
                print('Model unloaded.')
            except Exception as exc:
                print(f'Model unload check: {exc}', file=sys.stderr)
    args.home.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    output = args.home/f'disagreement-v4-seed{args.seed}-{stamp}.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nSaved results: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
