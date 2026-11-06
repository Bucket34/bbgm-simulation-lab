#!/usr/bin/env python3
"""BBGM Deck Agent V1: synthetic GM decision sandbox + read-only real BBGM export advisor.
Requires Python 3 standard library and an already running local Ollama instance.
Does not modify a Basketball GM league or execute arbitrary LLM output.
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import math
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home() / 'BBGMDeckAgentLab'
MODEL = 'qwen3:8b'
API = 'http://127.0.0.1:11435'


def make_dir():
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME


def ollama(prompt, model, base, ctx=4096):
    payload = {
        'model': model, 'prompt': prompt, 'stream': False,
        'think': False, 'format': 'json', 'keep_alive': '2m',
        'options': {'num_ctx': ctx, 'num_predict': 260, 'temperature': 0, 'seed': 42},
    }
    req = urllib.request.Request(
        base.rstrip('/') + '/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            out = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f'Cannot reach local Ollama at {base}: {exc}') from exc
    if out.get('done_reason') == 'length':
        raise RuntimeError('Model reply reached token limit; no decision accepted. Rerun or simplify prompt.')
    response = out.get('response', '')
    try:
        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Model did not produce valid JSON: {response[:300]!r}') from exc
    return result, {'input_tokens': out.get('prompt_eval_count'),
                    'output_tokens': out.get('eval_count'),
                    'tok_s': round(out.get('eval_count', 0) * 1e9 / max(1, out.get('eval_duration', 1)), 2)}


def unload(model, base):
    try:
        payload = json.dumps({'model': model, 'keep_alive': 0, 'stream': False}).encode()
        req = urllib.request.Request(base.rstrip('/') + '/api/generate', data=payload,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass


def player(pid, name, pos, ovr, tp, pss, diq, reb, age, cost):
    return {'id': pid, 'name': name, 'pos': pos, 'ovr': ovr, 'tp': tp,
            'pss': pss, 'diq': diq, 'reb': reb, 'age': age, 'cost': cost}


BASE_ROSTER = [
    player(1, 'Carter', 'PG', 61, 60, 74, 47, 31, 25, 5),
    player(2, 'Morris', 'SG', 64, 70, 55, 45, 39, 27, 6),
    player(3, 'Reed', 'SF', 59, 63, 52, 59, 48, 24, 4),
    player(4, 'Holt', 'PF', 62, 43, 39, 61, 72, 28, 5),
    player(5, 'Baker', 'C', 65, 22, 31, 67, 78, 29, 7),
    player(6, 'Lopez', 'G', 53, 66, 58, 42, 31, 22, 2),
    player(7, 'Diaz', 'F', 54, 51, 40, 61, 56, 23, 2),
]


def create_offers(rng, episode):
    # Generated fictional free-agent candidates. The fixed seed makes tests reproducible.
    archetypes = [
        ('Scorer', 'SG', 70, 79, 42, 35, 36, 8),
        ('Defender', 'SF', 64, 48, 48, 82, 58, 6),
        ('Playmaker', 'PG', 63, 57, 83, 47, 39, 6),
        ('Big', 'C', 68, 20, 31, 71, 87, 9),
    ]
    rng.shuffle(archetypes)
    offers = []
    for i, (name, pos, ovr, tp, pss, diq, reb, cost) in enumerate(archetypes):
        jitter = rng.randint(-3, 3)
        offers.append(player('ABCD'[i], f'{name}-{episode}', pos,
                             ovr + jitter, max(1, tp + rng.randint(-3, 3)),
                             pss, diq, reb, 22 + rng.randint(0, 10), cost))
    return offers


def strength(roster):
    starters = sorted(roster, key=lambda p: p['ovr'], reverse=True)[:5]
    avg = lambda field: sum(p[field] for p in starters) / len(starters)
    score = (avg('ovr') + 0.12 * (avg('tp') - 55) +
             0.11 * (avg('diq') - 55) + 0.10 * (avg('pss') - 55) +
             0.08 * (avg('reb') - 55))
    positions = [p['pos'] for p in starters]
    if any(p in positions for p in ('PG', 'G')) and any(p in positions for p in ('C', 'PF')):
        score += 1.2
    return score


def season_wins(roster, draws):
    # Toy game engine, NOT BBGM's real simulation algorithm.
    prob = max(0.10, min(0.9, 0.5 + (strength(roster) - 62) * 0.023))
    return sum(x < prob for x in draws)


def run_demo(args):
    make_dir()
    rng = random.Random(args.seed)
    results = []
    memory_file = HOME / 'agent_memory.json'
    try:
        memories = json.loads(memory_file.read_text()) if memory_file.exists() else []
        if not isinstance(memories, list):
            memories = []
    except (OSError, json.JSONDecodeError):
        memories = []
    previous_memory_count = len(memories)
    print('\nBBGM Deck GM Agent V1 — synthetic decision arena')
    print('NOTE: This is NOT the Basketball GM simulation engine or your saved league.')
    print(f'Prior synthetic decisions in persistent memory: {previous_memory_count}\n')
    for episode in range(1, args.trials + 1):
        offers = create_offers(rng, episode)
        draws = [rng.random() for _ in range(90)]
        by_id = {p['id']: p for p in offers}
        baseline = max(offers, key=lambda x: (x['ovr'], -x['cost']))
        prompt = ("You are the general manager of a fictional basketball team. Choose exactly "
                  "one free agent (A, B, C, or D) to improve next season's 90-game win total. "
                  "Prioritize lineup fit, defense, playmaking, shooting, and overall talent. "
                  "This is a toy simulator, not actual Basketball GM. All offers are affordable. "
                  "Output ONLY a JSON object with keys choice (A/B/C/D) and reason (max 30 words). "
                  "Do not claim you know the simulator's hidden scoring formula.\n\n"
                  f"Current roster: {json.dumps(BASE_ROSTER, separators=(',', ':'))}\n"
                  f"Available free agents: {json.dumps(offers, separators=(',', ':'))}\n"
                  f"Previous trials and outcomes: {json.dumps(memories[-3:], separators=(',', ':'))}")
        if args.offline:
            # Offline smoke test only. Not a test of an LLM decision.
            choice = baseline['id']
            reasoning = 'Offline baseline selects highest OVR.'
            stats = {}
        else:
            reply, stats = ollama(prompt, args.model, args.base, 4096)
            choice = str(reply.get('choice', '')).strip().upper()
            reasoning = str(reply.get('reason', ''))[:250]
            if choice not in by_id:
                print(f'Invalid agent choice {choice!r}; no action taken. Stopping.')
                return 2
        selected = by_id[choice]
        ai_team = BASE_ROSTER + [selected]
        baseline_team = BASE_ROSTER + [baseline]
        ai_wins = season_wins(ai_team, draws)
        baseline_wins = season_wins(baseline_team, draws)
        record = {'trial': episode, 'choice': choice, 'name': selected['name'],
                  'baseline_choice': baseline['id'], 'baseline_name': baseline['name'],
                  'agent_wins': ai_wins, 'baseline_wins': baseline_wins,
                  'advantage': ai_wins - baseline_wins, 'reason': reasoning, **stats}
        results.append(record)
        memory = {'trial': episode, 'chosen': selected['name'], 'wins': ai_wins,
                  'baseline_choice': baseline['name'], 'baseline_wins': baseline_wins,
                  'lesson': 'Compare both outcomes; one trial cannot prove a strategy is superior.'}
        memories.append(memory)
        print(f'Trial {episode}: Agent chose {choice} {selected["name"]} — {ai_wins} wins; '
              f'OVR baseline chose {baseline["id"]} {baseline["name"]} — {baseline_wins} wins')
        print(f'  Reason: {reasoning}')
        if stats:
            print(f'  Speed: {stats["tok_s"]} tok/s')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    outfile = HOME / f'demo-{stamp}.json'
    outfile.write_text(json.dumps({'mode': 'synthetic_toy', 'seed': args.seed,
                                   'results': results, 'memories': memories}, indent=2))
    memory_file.write_text(json.dumps(memories[-100:], indent=2))
    print(f'\nTotal simulated wins: agent={sum(r["agent_wins"] for r in results)}, '
          f'baseline={sum(r["baseline_wins"] for r in results)}')
    print(f'Saved results: {outfile}')
    print(f'Experience history: {memory_file} (model weights unchanged)')
    return 0


def load_league(path):
    p = Path(path).expanduser()
    if not p.exists():
        raise RuntimeError(f'No league file found at: {p}')
    opener = gzip.open if p.suffix.lower() == '.gz' else open
    try:
        with opener(p, 'rt', encoding='utf-8') as fh:
            league = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Cannot read exported league JSON: {exc}') from exc
    if not isinstance(league.get('teams'), list) or not isinstance(league.get('players'), list):
        raise RuntimeError('League export needs both teams and players selected under Tools > Export League.')
    return league


def full_team(team):
    return f'{team.get("region", "")} {team.get("name", "")}'.strip()


def latest_rating(p):
    ratings = p.get('ratings') or []
    if not isinstance(ratings, list) or not ratings:
        return {}
    return ratings[-1] if isinstance(ratings[-1], dict) else {}


def player_snapshot(p, season):
    r = latest_rating(p)
    born = p.get('born') or {}
    year = born.get('year') if isinstance(born, dict) else None
    age = (season - year) if isinstance(season, int) and isinstance(year, int) else None
    contract = p.get('contract') or {}
    return {'pid': p.get('pid'), 'name': ' '.join(x for x in (p.get('firstName'), p.get('lastName')) if x),
            'pos': p.get('pos'), 'age': age, 'ovr': r.get('ovr'), 'pot': r.get('pot'),
            'tp': r.get('tp'), 'pss': r.get('pss'), 'diq': r.get('diq'), 'reb': r.get('reb'),
            'contract_amount_raw': contract.get('amount') if isinstance(contract, dict) else None}


def detect_season(league):
    ga = league.get('gameAttributes') or {}
    if isinstance(ga, dict):
        s = ga.get('season')
        if isinstance(s, int):
            return s
        if isinstance(s, list) and s and isinstance(s[-1], dict):
            v = s[-1].get('value')
            if isinstance(v, int):
                return v
    return league.get('startingSeason') if isinstance(league.get('startingSeason'), int) else None


def run_league(args):
    league = load_league(args.league)
    teams = sorted(league['teams'], key=lambda t: str(full_team(t)))
    if not args.team:
        print('Found teams. Rerun with --team followed by a team name or numeric ID:')
        for t in teams:
            print(f'  {str(t.get("tid")):>3}  {full_team(t)}')
        return 0
    needle = args.team.strip().lower()
    matches = [t for t in teams if str(t.get('tid')) == needle or full_team(t).lower() == needle or
               (t.get('abbrev') or '').lower() == needle]
    if len(matches) != 1:
        raise RuntimeError(f'Team {args.team!r} not uniquely found. Run without --team to list available teams.')
    team = matches[0]
    tid = team.get('tid')
    season = detect_season(league)
    roster = [player_snapshot(p, season) for p in league['players'] if p.get('tid') == tid]
    free_agents = [player_snapshot(p, season) for p in league['players'] if p.get('tid') == -1]
    roster.sort(key=lambda p: (p['ovr'] if isinstance(p['ovr'], (int, float)) else -1), reverse=True)
    free_agents.sort(key=lambda p: (p['ovr'] if isinstance(p['ovr'], (int, float)) else -1), reverse=True)
    if not roster:
        raise RuntimeError('No roster players identified. Check the league export or team selection.')
    evidence = {'league_source': 'User-provided Basketball GM export',
                'team': full_team(team), 'tid': tid, 'season': season,
                'roster': roster[:20], 'free_agents_top': free_agents[:12],
                'total_free_agents': len(free_agents)}
    make_dir()
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    dest = HOME / f'bbgm-analysis-{stamp}.json'
    output = {'evidence': evidence, 'warning': 'Read-only advice. Not applied to Basketball GM.'}
    print(f'League loaded: {evidence["team"]}, roster={len(roster)}, free_agents={len(free_agents)}')
    print(f'Input is a COPY; no BBGM game data will be modified.')
    if args.offline:
        print('Offline extraction complete; no LLM contacted.')
    else:
        prompt = ('You are an assistant to a Basketball GM human general manager. Use ONLY the roster '
                  'and free-agent shortlist from this league export. Make a cautious assessment, '
                  'without inventing player details. Contract amounts are unnormalized raw export units; '
                  'do not assume cap room or affordability. Do not assert a free agent can be signed. '
                  'Reply with a JSON object containing: team_assessment (<=45 words), '
                  'suggested_free_agent_pid (must be one of listed free agent pids, or null), '
                  'reason (<=45 words), risks (<=40 words), next_manual_step (<=25 words). '
                  'You are advisory only: no trades or signings will occur.\n'
                  + json.dumps(evidence, separators=(',', ':')))
        reply, stats = ollama(prompt, args.model, args.base, 8192)
        candidate = reply.get('suggested_free_agent_pid')
        allowed = {str(p['pid']) for p in free_agents[:12] if p['pid'] is not None}
        if candidate is not None and str(candidate) not in allowed:
            reply['suggested_free_agent_pid'] = None
            reply['validation_warning'] = 'Suggested player ID not in supplied shortlist; rejected.'
        output['recommendation'] = reply
        output['metrics'] = stats
        print('\n--- Qwen3 8B GM recommendation (not executed) ---')
        print(json.dumps(reply, indent=2, ensure_ascii=False))
        print(f'Generation speed: {stats["tok_s"]} tok/s')
    dest.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved report: {dest}')
    return 0


def main():
    parser = argparse.ArgumentParser(description='Deck-only BBGM GM agent (toy demo or real export read-only advisor)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo', action='store_true', help='Synthetic toy seasons (NOT real BBGM engine)')
    mode.add_argument('--league', metavar='PATH', help='Actual BBGM Tools > Export League JSON or JSON.GZ')
    parser.add_argument('--team', help='Team name, abbrev, or numeric tid for --league')
    parser.add_argument('--trials', type=int, default=3, help='Toy trial count 1..10 (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Repeatable synthetic trial seed')
    parser.add_argument('--offline', action='store_true', help='Test deterministic logic; do not invoke Ollama')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--base', default=API)
    args = parser.parse_args()
    if not 1 <= args.trials <= 10:
        parser.error('--trials must be between 1 and 10')
    try:
        if args.demo:
            result = run_demo(args)
        else:
            result = run_league(args)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        result = 1
    finally:
        if not args.offline:
            unload(args.model, args.base)
    return result


if __name__ == '__main__':
    sys.exit(main())
