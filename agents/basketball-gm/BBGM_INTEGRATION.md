# Phase 4 — Real Basketball GM export reader (read-only)

**Scope:** `gm_core/bbgm_export.py` and `bbgm_inspect_export.py` read exported league JSON files to inspect teams and players. This is the first BBGM *data* adapter, **not** a connection to a running Basketball GM game, not an engine simulator, and not an AI general manager yet.

Official format references:
- https://basketball-gm.com/manual/customization/
- https://basketball-gm.com/manual/customization/players/
- https://basketball-gm.com/manual/customization/teams/

## Run tests

From the repository root:

```bash
python3 -m unittest discover -s agents/basketball-gm/tests -v
```

## Inspect a league export

1. On the computer/browser containing your league, open **Tools > Export League**. Select **Teams** and **Players** at minimum. Export a copy (do not overwrite your working league).
2. Copy the resulting `.json` or `.json.gz` file to the Steam Deck (e.g. `~/Downloads`). Do not commit the actual league into the repository.
3. From `agents/basketball-gm` run:

```bash
python3 bbgm_inspect_export.py ~/Downloads/YOUR_BBGM_LEAGUE.json.gz
python3 bbgm_inspect_export.py ~/Downloads/YOUR_BBGM_LEAGUE.json.gz --team 0 --players 10
```

The first command prints the counts and rosters for each team; the second prints player summaries from one team. Neither command writes files or changes the BBGM game state.

## Limitations / next stages

- Some customized league files omit fields; missing `ovr`, `pot`, `pos`, contracts, or ratings remain unknown, never fabricated.
- The last ratings entry from the export is shown, which might not match the current league season in a partial/custom export.
- No legality, negotiation, salary cap, trade, or simulation claims are made from these snapshot fields.
- Very large league exports are fully parsed into memory. On a Steam Deck, export only the needed components (Players/Teams) at first.
- This does not create or update live Basketball GM databases. Future action execution needs a separate, permissions-controlled game adapter and rollback plan.
- Keep exported leagues and report outputs out of Git. This code only prints information.
