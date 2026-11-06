#!/usr/bin/env python3
"""BBGM Deck Agent V1: paired evaluation on unseen synthetic situations.

Run from same Downloads folder as bbgm_deck_agent_v1.py.
Compares Qwen3 decisions WITHOUT versus WITH a frozen snapshot of prior memory.
Neither this evaluator nor the model modifies the original persistent memory.
Uses the original toy arena, NOT the real Basketball GM engine.
"""
import argparse
import datetime as dt
import importlib.util
import json
import random
import sys
from pathlib import Path

AGENT_FILE = Path(__file__).resolve().with_name('bbgm_deck_agent_v1.py')


def load_arena():
    if not AGENT_FILE.exists():
        raise RuntimeError(f'Missing original file: {AGENT_FILE}\nPut both scripts in ~/Downloads.')
    spec = importlib.util.spec_from_file_location('deck_bbgm_v1_arena', AGENT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load the original simulation functions.')
    arena = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arena)
    return arena


def get_memory(home):
    path = home / 'agent_memory.json'
    if not path.exists():
        raise RuntimeError(f'No prior-experience file at {path}. Run V1 trials first.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or len(data) < 3:
        raise RuntimeError('At least three recorded prior decisions are needed for this comparison.')
    return data[-6:]


def build_prompt(roster, offers, prior):
    return (
        'You are general manager of a fictional basketball team. Choose exactly one '
        'free agent (A, B, C, or D) to improve next season\'s 90-game win total. '
        'Prioritize lineup fit, defense, playmaking, shooting, and overall talent. '
        'This is a toy simulator, not actual Basketball GM. All offers are affordable. '
        'Output ONLY a JSON object with keys choice (A/B/C/D) and reason (max 30 words). '
        'Do not claim you know the simulator\'s hidden scoring formula. '
        'Prior results, if present, are from DIFFERENT earlier scenarios; no outcome '
        'is known for the current offers.\n\n'
        f'Current roster: {json.dumps(roster, separators=(",", ":"))}\n'
        f'Available free agents: {json.dumps(offers, separators=(",", ":"))}\n'
        f'Previous trials and outcomes: {json.dumps(prior, separators=(",", ":"))}'
    )


def trial_case(arena, rng, episode):
    offers = arena.create_offers(rng, episode)
    draws = [rng.random() for _ in range(90)]
    wins = {p['id']: arena.season_wins(arena.BASE_ROSTER + [p], draws) for p in offers}
    baseline = max(offers, key=lambda p: (p['ovr'], -p['cost']))
    oracle = max(offers, key=lambda p: (wins[p['id']], -p['cost']))
    return offers, wins, baseline, oracle


def run(args):
    arena = load_arena()
    home = Path.home() / 'BBGMDeckAgentLab'
    memory = get_memory(home)
    rng = random.Random(args.seed)
    cases = [trial_case(arena, rng, ep) for ep in range(1, args.trials + 1)]
    rows = []
    print('\nBBGM Deck GM V1 — Unseen-scenario memory evaluation')
    print('Synthetic toy engine; no Basketball GM league touched.')
    print(f'Test seed: {args.seed}; prior memory entries frozen: {len(memory)}')
    print('Two independent decisions on each identical scenario: without / with memory.')
    print('This evaluation will NOT update agent_memory.json.\n', flush=True)
    try:
        for ep, (offers, wins, baseline, oracle) in enumerate(cases, 1):
            by_id = {p['id']: p for p in offers}
            row = {'trial': ep, 'offers': offers, 'wins_for_each_offer': wins,
                   'baseline': {'choice': baseline['id'], 'wins': wins[baseline['id']]},
                   'best_available': {'choice': oracle['id'], 'wins': wins[oracle['id']]}}
            print(f'Trial {ep} (unseen offers):', flush=True)
            # Alternate order to avoid testing the memory condition second every time.
            conditions = ['without_memory', 'with_memory'] if ep % 2 else ['with_memory', 'without_memory']
            for condition in conditions:
                prior = memory if condition == 'with_memory' else []
                if args.offline:
                    choice = baseline['id']
                    reason = 'Offline control: choose highest OVR, no LLM call.'
                    stats = {}
                else:
                    response, stats = arena.ollama(
                        build_prompt(arena.BASE_ROSTER, offers, prior), args.model, args.base, 4096)
                    choice = str(response.get('choice', '')).strip().upper()
                    reason = str(response.get('reason', ''))[:250]
                    if choice not in by_id:
                        raise RuntimeError(f'Trial {ep}, {condition}: invalid choice {choice!r}; stopped.')
                row[condition] = {'choice': choice, 'player': by_id[choice]['name'],
                                  'wins': wins[choice], 'reason': reason, **stats}
                label = 'With memory' if condition == 'with_memory' else 'No memory  '
                print(f'  {label}: {choice} {by_id[choice]["name"]}, '
                      f'{wins[choice]} wins' + (f', {stats["tok_s"]} tok/s' if stats else ''), flush=True)
            delta = row['with_memory']['wins'] - row['without_memory']['wins']
            print(f'  OVR baseline: {baseline["name"]}, {wins[baseline["id"]]} wins; '
                  f'best of 4: {oracle["name"]}, {wins[oracle["id"]]} wins; '
                  f'memory difference: {delta:+d}\n', flush=True)
            rows.append(row)
    finally:
        if not args.offline:
            arena.unload(args.model, args.base)

    totals = {k: sum(row[k]['wins'] for row in rows)
              for k in ('without_memory', 'with_memory', 'baseline', 'best_available')}
    print('=== TOTAL WINS (paired, unseen scenarios) ===')
    print(f'No memory:    {totals["without_memory"]}')
    print(f'With memory:  {totals["with_memory"]}')
    print(f'OVR baseline: {totals["baseline"]}')
    print(f'Best of four: {totals["best_available"]} (hindsight upper bound)')
    print(f'Memory effect: {totals["with_memory"] - totals["without_memory"]:+d} wins')
    if args.offline:
        print('Offline smoke test only: these are NOT AI decisions.')
    else:
        home.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = home / f'unseen-memory-eval-seed{args.seed}-{stamp}.json'
        path.write_text(json.dumps({'kind': 'paired_unseen_synthetic_test',
                                    'seed': args.seed, 'prior_training_memories': memory,
                                    'results': rows, 'totals': totals}, indent=2), encoding='utf-8')
        print(f'Results saved: {path}')
    print('Original memory unchanged. The model weights were never changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=99, help='Unseen scenarios; do not use prior training seed 42')
    parser.add_argument('--trials', type=int, default=3, choices=range(1, 11), metavar='1..10')
    parser.add_argument('--model', default='qwen3:8b')
    parser.add_argument('--base', default='http://127.0.0.1:11435')
    parser.add_argument('--offline', action='store_true', help='Test local logic with OVR baseline, no Ollama')
    args = parser.parse_args()
    if args.seed == 42:
        parser.error('Seed 42 was used for training/replay; choose a different seed, e.g. 99.')
    try:
        run(args)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as err:
        print(f'ERROR: {err}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
