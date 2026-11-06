"""Read-only Basketball GM / ZenGM league export snapshot adapter.

This deliberately does NOT simulate games, apply transactions, or write league files.
It reads a JSON or JSON.gz export and returns a small normalized snapshot.

Documentation: https://basketball-gm.com/manual/customization/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gzip
import json


class LeagueExportError(ValueError):
    """The supplied file is not a supported/usable league export."""


@dataclass(frozen=True)
class Team:
    tid: int
    name: str
    abbreviation: str


@dataclass(frozen=True)
class Player:
    pid: int | None
    name: str
    tid: int
    position: str | None
    season: int | None
    overall: int | None
    potential: int | None
    attributes: dict[str, int]
    contract_amount: int | float | None
    contract_expires: int | None


@dataclass(frozen=True)
class LeagueSnapshot:
    version: int | None
    season: int | None
    teams: tuple[Team, ...]
    players: tuple[Player, ...]

    def roster(self, tid: int) -> tuple[Player, ...]:
        return tuple(p for p in self.players if p.tid == tid)

    def team(self, tid: int) -> Team | None:
        return next((t for t in self.teams if t.tid == tid), None)

    @property
    def free_agents(self) -> tuple[Player, ...]:
        return tuple(p for p in self.players if p.tid == -1)

    @property
    def prospects(self) -> tuple[Player, ...]:
        return tuple(p for p in self.players if p.tid == -2)

    @property
    def retired(self) -> tuple[Player, ...]:
        return tuple(p for p in self.players if p.tid == -3)


def _integer_or_none(value: Any) -> int | None:
    # Reject bool: bool is technically a subclass of int in Python.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _season_from_attributes(data: dict[str, Any]) -> int | None:
    attrs = data.get("gameAttributes")
    if isinstance(attrs, dict):
        return _integer_or_none(attrs.get("season"))
    if isinstance(attrs, list):
        for entry in attrs:
            if isinstance(entry, dict) and entry.get("key") == "season":
                return _integer_or_none(entry.get("value"))
    return _integer_or_none(data.get("startingSeason"))


def _nonempty_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _make_player(raw: dict[str, Any], number: int) -> Player:
    tid = _integer_or_none(raw.get("tid"))
    if tid is None:
        raise LeagueExportError(f"players[{number}].tid must be an integer")
    ratings = raw.get("ratings", [])
    if not isinstance(ratings, list):
        raise LeagueExportError(f"players[{number}].ratings must be a list if present")
    # The final ratings record in the export, not a forecast or a made-up season.
    latest = ratings[-1] if ratings else {}
    if not isinstance(latest, dict):
        raise LeagueExportError(f"players[{number}].ratings[-1] must be an object")
    attrs = {
        k: v for k, v in latest.items()
        if k in {"hgt", "stre", "spd", "jmp", "endu", "ins", "dnk", "ft",
                 "fg", "tp", "oiq", "diq", "drb", "pss", "reb"}
        and _integer_or_none(v) is not None
    }
    contract = raw.get("contract")
    if not isinstance(contract, dict):
        contract = {}
    amount = contract.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        amount = None
    first = _nonempty_str(raw.get("firstName"))
    last = _nonempty_str(raw.get("lastName"))
    display_name = " ".join(x for x in (first, last) if x) or f"Player {number}"
    return Player(
        pid=_integer_or_none(raw.get("pid")), name=display_name, tid=tid,
        position=_nonempty_str(latest.get("pos")) or _nonempty_str(raw.get("pos")) or None,
        season=_integer_or_none(latest.get("season")),
        overall=_integer_or_none(latest.get("ovr")),
        potential=_integer_or_none(latest.get("pot")),
        attributes=attrs,
        contract_amount=amount,
        contract_expires=_integer_or_none(contract.get("exp")),
    )


def normalize_export(data: Any) -> LeagueSnapshot:
    """Convert parsed league JSON to immutable-ish data transfer objects.

    Does not assert a complete/restorable BBGM league or financial legality.
    """
    if not isinstance(data, dict):
        raise LeagueExportError("League export root must be a JSON object")
    teams_raw, players_raw = data.get("teams"), data.get("players")
    if not isinstance(teams_raw, list) or not isinstance(players_raw, list):
        raise LeagueExportError("Expected teams and players arrays; export both in BBGM")
    teams = []
    seen_tids = set()
    for i, t in enumerate(teams_raw):
        if not isinstance(t, dict):
            raise LeagueExportError(f"teams[{i}] must be an object")
        tid = _integer_or_none(t.get("tid"))
        if tid is None or tid < 0 or tid in seen_tids:
            raise LeagueExportError(f"teams[{i}].tid must be a unique nonnegative integer")
        seen_tids.add(tid)
        region, name = _nonempty_str(t.get("region")), _nonempty_str(t.get("name"))
        teams.append(Team(tid=tid, name=(" ".join(x for x in (region, name) if x) or f"Team {tid}"),
                          abbreviation=_nonempty_str(t.get("abbrev"))))
    players = []
    seen_pids = set()
    for i, p in enumerate(players_raw):
        if not isinstance(p, dict):
            raise LeagueExportError(f"players[{i}] must be an object")
        player = _make_player(p, i)
        if player.tid >= 0 and player.tid not in seen_tids:
            raise LeagueExportError(f"players[{i}].tid {player.tid} has no matching team")
        if player.pid is not None:
            if player.pid in seen_pids:
                raise LeagueExportError(f"Duplicate player pid {player.pid}")
            seen_pids.add(player.pid)
        players.append(player)
    return LeagueSnapshot(
        version=_integer_or_none(data.get("version")),
        season=_season_from_attributes(data),
        teams=tuple(teams), players=tuple(players),
    )


def load_export(path: str | Path) -> LeagueSnapshot:
    """READ ONLY. Supports BBGM .json and .json.gz exports; never writes files."""
    file = Path(path).expanduser()
    suffixes = file.suffixes
    if suffixes[-1:] != [".json"] and suffixes[-2:] != [".json", ".gz"]:
        raise LeagueExportError("Expected a .json or .json.gz Basketball GM export")
    try:
        if file.suffix == ".gz":
            with gzip.open(file, "rt", encoding="utf-8-sig") as src:
                raw = json.load(src)
        else:
            with file.open("r", encoding="utf-8-sig") as src:
                raw = json.load(src)
    except (OSError, UnicodeError, json.JSONDecodeError, EOFError) as exc:
        raise LeagueExportError(f"Unable to read league export: {exc}") from exc
    return normalize_export(raw)
