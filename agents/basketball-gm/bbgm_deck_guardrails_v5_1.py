#!/usr/bin/env python3
"""BBGM Deck V5.1 — audited multi-season decision controller.

Runs the exact toy league engine from bbgm_deck_multiseason_v5.py.
Three separate Qwen3 closed-loop management modes on identical initial episodes:
  alone   - independent Qwen choice, validated for legality
  advised - Qwen sees frozen Python policy estimates and may disagree
  guarded - Qwen proposes; an external rule can override it using estimated value
Also runs greedy and frozen Python policy baselines.

The policy is NOT Qwen fine-tuning. The guardrail is not a true oracle and may
make worse decisions; only subsequent held-out evaluation can tell.

Read-only inference: no model-generated code or shell instructions are executed.
No BBGM leagues, prior results, training weights, or original V5 files changed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import bbgm_deck_multiseason_v5 as sim
except ImportError as exc:
    raise SystemExit(
        "Could not import bbgm_deck_multiseason_v5.py. Save both files "
        "in the same folder (e.g. ~/Downloads)."
    ) from exc


# Keep the old public function names available for existing experiments/tests.
from gm_core import policy_pick, rationale_flags, resolve_decision
from gm_core.llm import llm_choice
from gm_core.ollama import verify_gpu, unload_model


def play(episode, params, mode, margin, model, base, ai_choice=llm_choice):
    assert mode in ('alone', 'advised', 'guarded')
    signings = []
    history = []
    details = []
    total = 0.0
    override_count = 0
    rejected_count = 0
    flagged_count = 0
    for year in range(episode['years']):
        legal = sim.available_offers(episode, signings, year)
        legal_by_id = {p['id']: p for p in legal}
        cap_left = episode['cap'] - sum(p['cost'] for p in signings if p['ends'] > year)
        recommended, scores = policy_pick(episode, signings, year, params)
        advice = scores if mode == 'advised' else None
        proposal = ai_choice(episode, signings, history, year, model, base, advice)
        decision = resolve_decision(proposal, legal_by_id, recommended, scores, mode, margin)
        proposal_id = decision.proposal_id
        invalid = decision.invalid
        chosen = decision.final_action
        override_reason = decision.override_reason
        estimated_gain = decision.estimated_gain
        if invalid:
            rejected_count += 1
        if mode == 'guarded' and chosen != decision.effective_proposal:
            override_count += 1
        reason = str(proposal.get('reason', ''))
        flags = rationale_flags(reason, scores, cap_left)
        flagged_count += len(flags)
        if chosen != 'PASS':
            if chosen not in legal_by_id:
                raise RuntimeError(f'Controller selected illegal offer {chosen}')
            signings.append(legal_by_id[chosen])
        active = [p for p in signings if p['ends'] > year]
        wins = sim.expected_wins(episode['initial'] + active, year)
        total += wins
        details.append({
            'season': year + 1,
            'cap_before': cap_left,
            'legal_choices': ['PASS'] + list(legal_by_id),
            'qwen_proposal': proposal_id,
            'qwen_invalid': invalid,
            'qwen_reason': reason,
            'qwen_finish': proposal.get('finish'),
            'policy_recommendation': recommended,
            'policy_scores': {k: round(v, 3) for k, v in sorted(scores.items())},
            'estimated_policy_advantage': estimated_gain,
            'final_action': chosen,
            'overridden': bool(override_reason),
            'override_reason': override_reason,
            'rationale_flags': flags,
            'expected_wins': round(wins, 3),
            'cap_after': episode['cap'] - sum(p['cost'] for p in active),
        })
        history.append({'season': year + 1, 'choice': chosen,
                        'expected_wins': round(wins, 3)})
        if override_reason:
            print(f'    Controller: {override_reason}', flush=True)
        if flags:
            print('    Rationale warning: ' + '; '.join(flags), flush=True)
    return {'total_wins': total, 'seasons': details, 'overrides': override_count,
            'invalid_proposals': rejected_count, 'rationale_flags': flagged_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=37001,
                        help='The existing V5 seed (test seed is seed + 100000003)')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--start-episode', type=int, default=1,
                        help='1-based V5 held-out episode number (default 1)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Predicted policy-score advantage needed for an override')
    parser.add_argument('--discount', type=float, default=0.7)
    parser.add_argument('--cap-penalty', type=float, default=1.5)
    parser.add_argument('--signing-threshold', type=float, default=1.5)
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--base', default='http://127.0.0.1:11435')
    parser.add_argument('--home', type=Path,
                        default=Path.home() / 'BBGMDeckAgentLab')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 5:
        parser.error('--episodes must be 1 to 5')
    if not 1 <= args.start_episode <= 100:
        parser.error('--start-episode must be 1 to 100')
    if not 0 <= args.margin <= 50:
        parser.error('--margin must be 0 to 50')
    params = (args.discount, args.cap_penalty, args.signing_threshold)
    test_seed = args.seed + 100000003
    episodes = [sim.make_episode(test_seed, args.start_episode - 1 + idx, 3)
                for idx in range(args.episodes)]
    print('BBGM Deck V5.1 — AUDITED MULTI-SEASON GM', flush=True)
    print('Exact V5 synthetic simulation; no actual Basketball GM engine or league touched.')
    print('Qwen-alone, Qwen-advised, Qwen-guarded, greedy and learned Python baseline.')
    print(f'V5 params: {params}; guard threshold: {args.margin:+.2f} predicted score', flush=True)
    print('Note: modes start identically but can have different rosters after decisions.')
    report = {
        'experiment': 'V5.1 guarded vs advisory synthetic multi-season',
        'settings': {'seed': args.seed, 'test_seed': test_seed, 'episodes': args.episodes,
                     'seasons': 3, 'start_episode': args.start_episode,
                     'margin': args.margin, 'params': params,
                     'model': args.model},
        'episodes': []
    }
    totals = {k: 0.0 for k in ['alone', 'advised', 'guarded', 'greedy', 'learned']}
    try:
        verify_gpu(args.base, args.model)
        for number, episode in enumerate(episodes, 1):
            print(f'\n=== V5 Episode {args.start_episode + number - 1} ===', flush=True)
            results = {}
            for mode in ('alone', 'advised', 'guarded'):
                print(f'  {mode.capitalize()}:', flush=True)
                res = play(episode, params, mode, args.margin, args.model, args.base)
                results[mode] = res
                totals[mode] += res['total_wins']
                actions = ' / '.join(s['final_action'] for s in res['seasons'])
                print(f'    Choices: {actions}; mean {res["total_wins"]/3:.2f} wins/season; '
                      f'overrides={res["overrides"]}; invalid={res["invalid_proposals"]}', flush=True)
            greedy, gh = sim.run_episode(episode, sim.choose_greedy)
            learned, lh = sim.run_episode(episode,
                            lambda e, s, y: sim.choose_learned(e, s, y, params))
            totals['greedy'] += greedy
            totals['learned'] += learned
            results['greedy'] = {'total_wins': greedy, 'seasons': gh}
            results['learned'] = {'total_wins': learned, 'seasons': lh}
            print(f'  Greedy: {greedy/3:.2f}; Learned Python: {learned/3:.2f}', flush=True)
            report['episodes'].append({'id': episode['id'], 'modes': results})
        print('\n=== Mean expected wins per 90-game season ===')
        for mode in ('alone', 'advised', 'guarded', 'greedy', 'learned'):
            print(f'  {mode:9}: {totals[mode]/(args.episodes*3):.3f}')
        report['averages'] = {k: v / (args.episodes*3) for k, v in totals.items()}
    except Exception as exc:
        report['error'] = str(exc)
        print(f'Experiment stopped safely: {exc}', file=sys.stderr, flush=True)
    finally:
        try:
            unload_model(args.base, args.model)
            print('Model unloaded.', flush=True)
        except Exception:
            print('Model unload could not be verified; check /api/ps.', flush=True)
        args.home.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = args.home / f'guardrails-v51-{args.seed}-{stamp}.json'
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report saved: {path}', flush=True)
    if 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
