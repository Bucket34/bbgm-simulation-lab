#!/usr/bin/env python3
"""BBGM Deck advisor evaluation V3.1 -- legal-action and diagnostic fixes.

Compares Qwen3 alone, Qwen3 with learned-policy advice, the policy itself,
and a conservative hybrid that overrides Qwen3 only when the policy predicts
at least a configurable improvement. No Basketball GM engine is used.

Requirements: Python 3 standard library, bbgm_deck_learning_v2.py in the same
folder, ~/BBGMDeckAgentLab/policy_v2.json, and Alpaca-managed Ollama running
on localhost:11435 with Vulkan acceleration.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import urllib.request
import urllib.error
from pathlib import Path

import bbgm_deck_learning_v2 as sim


def request_json(base, path, payload=None, timeout=240):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(
        base.rstrip('/') + path,
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.load(reply)


def unload(base, model):
    try:
        request_json(base, '/api/generate', {
            'model': model, 'stream': False, 'keep_alive': 0,
        }, timeout=45)
    except Exception as exc:
        print('Unload check:', exc)


def check_gpu(base, model):
    version = request_json(base, '/api/version', timeout=10).get('version', '?')
    request_json(base, '/api/generate', {
        'model': model, 'stream': False, 'keep_alive': '5m',
        'options': {'num_ctx': 8192},
    }, timeout=240)
    models = request_json(base, '/api/ps', timeout=10).get('models', [])
    found = next((m for m in models if m.get('name') == model), None)
    if not found:
        raise RuntimeError(f'{model} not listed as loaded by /api/ps')
    size = found.get('size', 0)
    vram = found.get('size_vram', 0)
    print(f'Ollama {version}; GPU allocation {vram:,}/{size:,} bytes')
    if not size or size != vram:
        raise RuntimeError('Full GPU offloading not confirmed. Stop to avoid CPU-only benchmarking.')


def make_decision(case, estimates, model, base):
    """Offer ONLY legal actions to the LLM; never disclose oracle outcomes."""
    legal = sim.legal_offers(case)
    display_case = dict(case, offers=legal)
    allowed = ', '.join(p['id'] for p in legal)
    prompt = (
        f'Valid choices for this scenario: {allowed}. All listed offers are '
        'affordable. Return exactly one of these IDs as JSON. '
        'An invalid choice will be rejected, not silently corrected.\n'
        + sim.llm_prompt(display_case, estimates)
    )
    return sim.query_llm(prompt, model, base)


def score_choice(case, choice):
    """An invalid action is rejected as NO SIGNING, not counted as a valid pick."""
    if choice is None:
        return sim.expected_wins(case['roster'])
    return sim.expected_wins(case['roster']) + sim.oracle_results(case)[choice]


def label(choice):
    return choice if choice is not None else 'NONE (invalid)'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=int, default=5,
                    help='Number of new scenarios (1-30), two LLM calls each; default 5')
    ap.add_argument('--start-case', type=int, default=1,
                    help='One-based scenario number (default 1); use 5 to resume after first four')
    ap.add_argument('--seed', type=int, default=16001,
                    help='New scenario seed; avoid previously tested 9901')
    ap.add_argument('--margin', type=float, default=1.0,
                    help='Hybrid override threshold in predicted expected wins (default 1.0)')
    ap.add_argument('--model', default='qwen3:8b')
    ap.add_argument('--base', default='http://127.0.0.1:11435')
    ap.add_argument('--home', type=Path, default=Path.home() / 'BBGMDeckAgentLab')
    args = ap.parse_args()
    if not 1 <= args.cases <= 30:
        ap.error('--cases must be between 1 and 30')
    if not 1 <= args.start_case <= 10000:
        ap.error('--start-case must be between 1 and 10000')
    if not math.isfinite(args.margin) or args.margin < 0:
        ap.error('--margin must be finite and nonnegative')
    if args.seed in (9901, 4201):
        ap.error('Choose a fresh seed (not 9901 for V2 evaluation or 4201 for training)')
    policy_path = args.home / 'policy_v2.json'
    if not policy_path.is_file():
        ap.error('Missing policy_v2.json; first run bbgm_deck_learning_v2.py')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    if policy.get('feature_names') != list(sim.FEATURE_NAMES):
        ap.error('The saved policy features do not match the V2 simulator')
    weights = policy['weights']
    if len(weights) != len(sim.FEATURE_NAMES):
        ap.error('The saved policy weights have an invalid length')

    print('BBGM Deck Advisor Evaluation V3.1 -- SYNTHETIC simulator ONLY', flush=True)
    print('No actual BBGM league, model weights, or previous memory will be changed.')
    print(f'New scenario seed: {args.seed}; cases: {args.cases}; first case: {args.start_case}; '
          f'hybrid advice threshold: {args.margin:.2f} predicted wins', flush=True)
    print('Five strategies: Qwen alone, Qwen with advice, Python policy, '
          'conservative hybrid, OVR baseline. Oracle is hindsight only.', flush=True)
    print('Hybrid threshold is a heuristic, NOT a calibrated confidence score.')
    print('Invalid LLM actions become NO SIGNING for that LLM condition; hybrid can use the legal policy fallback.\n', flush=True)

    cases = [sim.make_case(args.seed, i)
             for i in range(args.start_case - 1, args.start_case - 1 + args.cases)]
    rows = []
    try:
        check_gpu(args.base, args.model)
        for index, case in enumerate(cases):
            legal = sim.legal_offers(case)
            ids = {p['id'] for p in legal}
            estimates = {p['id']: round(sim.dot(weights, sim.features(case, p)), 3)
                         for p in legal}
            # Alternate order to reduce condition-order bias.
            order = [('alone', None), ('advised', estimates)]
            if index % 2:
                order.reverse()
            model_decisions = {}
            for condition, advice in order:
                chosen, reason, speed = make_decision(case, advice, args.model, args.base)
                invalid = chosen not in ids
                if invalid:
                    print(f'  REJECTED: Scenario {case["id"]+1}, {condition} returned '
                          f'{chosen!r}; legal IDs: {sorted(ids)}. '
                          'Scored as no signing.', flush=True)
                model_decisions[condition] = {
                    'choice': None if invalid else chosen,
                    'raw_choice': chosen,
                    'invalid': invalid,
                    'reason': reason,
                    'speed_tok_s': speed,
                }

            alone = model_decisions['alone']['choice']
            advised = model_decisions['advised']['choice']
            policy_choice = sim.learned_choice(case, weights)['id']
            ovr = sim.baseline_choice(case)['id']
            true_rewards = sim.oracle_results(case)
            oracle_choice = max(true_rewards, key=true_rewards.get)
            predicted_gain = estimates[policy_choice] - (estimates[alone] if alone else 0.0)
            # Invalid Qwen actions are not executed. The hybrid's independent
            # validator falls back to a legal policy action in that case.
            override = alone is None or predicted_gain >= args.margin
            hybrid = policy_choice if override else alone
            hybrid_reason = ('invalid-action fallback' if alone is None
                             else ('policy predicted benefit' if override else 'retain Qwen choice'))
            picks = {
                'qwen_alone': alone, 'qwen_advised': advised,
                'python_policy': policy_choice, 'hybrid': hybrid,
                'ovr_baseline': ovr, 'oracle': oracle_choice,
            }
            wins = {key: round(score_choice(case, value), 6)
                    for key, value in picks.items()}
            rows.append({
                'case_id': case['id'], 'budget': case['budget'],
                'legal_ids': sorted(ids), 'legal_count': len(ids),
                'policy_estimates': estimates, 'predicted_gain_over_qwen': round(predicted_gain, 3),
                'choices': picks, 'hybrid_reason': hybrid_reason,
                'expected_wins': wins, 'model_details': model_decisions,
            })
            print(f'\nScenario {case["id"]+1} (budget {case["budget"]}; '
                  f'legal: {", ".join(sorted(ids))}):', flush=True)
            print(f'  Qwen alone    {label(alone):<15} {wins["qwen_alone"]:.2f} wins', flush=True)
            print(f'  Qwen + advice {label(advised):<15} {wins["qwen_advised"]:.2f} wins', flush=True)
            print(f'  Python policy {policy_choice:<15} {wins["python_policy"]:.2f} wins', flush=True)
            print(f'  Hybrid        {hybrid:<15} {wins["hybrid"]:.2f} wins '
                  f'({hybrid_reason}, estimated gain {predicted_gain:+.2f})', flush=True)
            print(f'  OVR baseline  {ovr:<15} {wins["ovr_baseline"]:.2f} wins', flush=True)
            print(f'  Oracle        {oracle_choice:<15} {wins["oracle"]:.2f} wins', flush=True)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print('\nExperiment stopped:', exc, flush=True)
    finally:
        unload(args.base, args.model)

    if not rows:
        print('No completed scenarios; no report saved.')
        return 1
    print('\n=== RESULTS (new toy scenarios only) ===')
    print(f'Completed {len(rows)}/{args.cases} scenarios.')
    print(f'{"Strategy":<17} {"Mean wins":>10} {"Optimal picks":>14} {"Mean regret":>12}')
    keys = ('qwen_alone', 'qwen_advised', 'python_policy',
            'hybrid', 'ovr_baseline', 'oracle')
    summary = {}
    for key in keys:
        avg = sum(r['expected_wins'][key] for r in rows)/len(rows)
        mean_regret = sum(r['expected_wins']['oracle'] - r['expected_wins'][key]
                          for r in rows)/len(rows)
        correct = sum(r['expected_wins']['oracle'] - r['expected_wins'][key] < 1e-5
                      for r in rows)
        summary[key] = {'mean_expected_wins': avg, 'optimal_picks': correct,
                        'mean_regret': mean_regret}
        print(f'{key:<17} {avg:>10.2f} {correct:>9}/{len(rows):<4} {mean_regret:>12.3f}')

    helped = sum(r['expected_wins']['qwen_advised'] > r['expected_wins']['qwen_alone']+1e-5
                 for r in rows)
    hurt = sum(r['expected_wins']['qwen_advised'] < r['expected_wins']['qwen_alone']-1e-5
               for r in rows)
    unchanged = len(rows) - helped - hurt
    print(f'Advice helped in {helped}, hurt in {hurt}, unchanged in {unchanged} scenarios.')
    invalid_by_condition = {condition: sum(r['model_details'][condition]['invalid'] for r in rows)
                            for condition in ('alone', 'advised')}
    agreements = sum(len({r['choices'][k] for k in ('qwen_alone', 'qwen_advised',
                        'python_policy', 'hybrid', 'ovr_baseline')}) == 1 for r in rows)
    print(f'Rejected illegal responses: alone={invalid_by_condition["alone"]}, '
          f'advised={invalid_by_condition["advised"]}.')
    print(f'All five strategies agreed on {agreements}/{len(rows)} cases.')
    print('Note: this threshold was not calibrated; all success is within the invented toy rules.')

    output_dir = args.home
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f'advisor-v31-seed{args.seed}-start{args.start_case}-{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
    out = output_dir/name
    out.write_text(json.dumps({
        'experiment': 'toy_bbgm_advisor_v3', 'settings': {
            'seed': args.seed, 'start_case': args.start_case, 'cases_requested': args.cases,
            'cases_completed': len(rows), 'margin': args.margin,
            'model': args.model, 'base': args.base,
        }, 'summary': summary, 'invalid_counts': invalid_by_condition, 'all_agree': agreements,
        'advice_helped': helped, 'advice_hurt': hurt,
        'advice_unchanged': unchanged, 'scenarios': rows,
    }, indent=2), encoding='utf-8')
    print(f'Results saved: {out}')
    return 0 if len(rows) == args.cases else 2


if __name__ == '__main__':
    raise SystemExit(main())
