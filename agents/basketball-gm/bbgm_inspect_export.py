#!/usr/bin/env python3
"""Read-only inspection of a real Basketball GM league export. No Ollama needed."""
from __future__ import annotations

import argparse
from gm_core.bbgm_export import load_export, LeagueExportError


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Basketball GM export inspector")
    parser.add_argument("export_file", help="Path to exported .json or .json.gz league file")
    parser.add_argument("--team", type=int, help="Display an individual team by tid")
    parser.add_argument("--players", type=int, default=5, help="Number of roster players to display (default 5)")
    args = parser.parse_args()
    if args.players < 0 or args.players > 100:
        parser.error("--players must be between 0 and 100")
    try:
        league = load_export(args.export_file)
    except LeagueExportError as exc:
        parser.exit(2, f"Export error: {exc}\n")
    print("BBGM EXPORT INSPECTOR — READ ONLY (no live game connection)")
    print(f"League version: {league.version if league.version is not None else 'not provided'}")
    print(f"Current season: {league.season if league.season is not None else 'not provided'}")
    print(f"Teams: {len(league.teams)} | Players: {len(league.players)}")
    print(f"Free agents: {len(league.free_agents)} | Prospects: {len(league.prospects)} | Retired: {len(league.retired)}")
    selected = league.teams if args.team is None else tuple(t for t in league.teams if t.tid == args.team)
    if args.team is not None and not selected:
        parser.exit(2, f"No team with tid {args.team}\n")
    print("\nTEAM ROSTERS")
    for team in sorted(selected, key=lambda x: x.tid):
        roster = league.roster(team.tid)
        print(f"  {team.tid:>3}: {team.name} ({team.abbreviation or '-'}) — {len(roster)} players")
        if args.team is not None and args.players:
            # Display highest available OVR first, unknown OVR last. No fabricated ratings.
            for p in sorted(roster, key=lambda p: (p.overall is not None, p.overall or 0), reverse=True)[:args.players]:
                print(f"      {p.name}: {p.position or '?'} OVR={p.overall if p.overall is not None else '?'}"
                      f" POT={p.potential if p.potential is not None else '?'}"
                      f" [pid={p.pid if p.pid is not None else '?'}]")
    print("\nNo files changed. BBGM simulator was not launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
