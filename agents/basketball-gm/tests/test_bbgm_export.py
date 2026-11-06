"""Read-only adapter tests. Uses invented fixtures, never alters a real league."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gm_core.bbgm_export import LeagueExportError, load_export, normalize_export


def fixture():
    return {
        "version": 66,
        "gameAttributes": {"season": 2026},
        "teams": [
            {"tid": 0, "region": "Lagos", "name": "Eagles", "abbrev": "LAG"},
            {"tid": 1, "region": "Tokyo", "name": "Kaiju", "abbrev": "TOK"},
        ],
        "players": [
            {"pid": 14, "firstName": "Ada", "lastName": "Test", "tid": 0,
             "contract": {"amount": 12000, "exp": 2028},
             "ratings": [{"season": 2025, "ovr": 60, "tp": 55},
                         {"season": 2026, "ovr": 65, "pot": 77, "pos": "SG", "tp": 76, "diq": 50}]},
            {"pid": 15, "firstName": "B", "lastName": "Free", "tid": -1,
             "ratings": [{"season": 2026, "ovr": 58, "pos": "C"}]},
            {"pid": 16, "firstName": "C", "lastName": "Prospect", "tid": -2,
             "ratings": [{"tp": 45}]},
            {"pid": 17, "firstName": "D", "lastName": "Retired", "tid": -3},
            {"pid": 18, "firstName": "E", "lastName": "Bench", "tid": 1, "ratings": []},
        ],
    }


class BBGMExportTests(unittest.TestCase):
    def test_rosters_and_free_agent_categories(self):
        league = normalize_export(fixture())
        self.assertEqual(len(league.teams), 2)
        self.assertEqual([p.pid for p in league.roster(0)], [14])
        self.assertEqual([p.pid for p in league.free_agents], [15])
        self.assertEqual([p.pid for p in league.prospects], [16])
        self.assertEqual([p.pid for p in league.retired], [17])

    def test_latest_rating_and_contract_are_read_without_conversion(self):
        player = normalize_export(fixture()).roster(0)[0]
        self.assertEqual((player.position, player.overall, player.potential), ("SG", 65, 77))
        self.assertEqual((player.attributes["tp"], player.contract_amount, player.contract_expires), (76, 12000, 2028))
        self.assertEqual(player.season, 2026)

    def test_missing_ratings_do_not_invent_overall(self):
        player = normalize_export(fixture()).roster(1)[0]
        self.assertIsNone(player.overall)
        self.assertIsNone(player.position)

    def test_game_attributes_list_season(self):
        league = fixture()
        league["gameAttributes"] = [{"key": "season", "value": 2027}]
        self.assertEqual(normalize_export(league).season, 2027)

    def test_malformed_files_rejected(self):
        for raw in (None, [], {}, {"teams": {}, "players": []},
                    {"teams": [], "players": "not a list"}):
            with self.subTest(raw=raw), self.assertRaises(LeagueExportError):
                normalize_export(raw)

    def test_invalid_team_reference_rejected(self):
        raw = fixture()
        raw["players"][0]["tid"] = 999
        with self.assertRaises(LeagueExportError):
            normalize_export(raw)

    def test_duplicate_ids_rejected(self):
        raw = fixture()
        raw["players"][1]["pid"] = 14
        with self.assertRaises(LeagueExportError):
            normalize_export(raw)

    def test_plain_and_gzip_files_unchanged(self):
        original = json.dumps(fixture()).encode("utf-8")
        with tempfile.TemporaryDirectory() as temp:
            for filename, payload in (("league.json", original), ("league.json.gz", gzip.compress(original))):
                path = Path(temp) / filename
                path.write_bytes(payload)
                before = hashlib.sha256(path.read_bytes()).digest()
                snapshot = load_export(path)
                after = hashlib.sha256(path.read_bytes()).digest()
                self.assertEqual(before, after)
                self.assertEqual(len(snapshot.players), 5)

    def test_wrong_suffix_and_broken_json(self):
        with tempfile.TemporaryDirectory() as temp:
            for filename, payload in (("league.txt", b"{}"), ("league.json", b"{"),
                                      ("league.json.gz", b"not-gzip")):
                path = Path(temp) / filename
                path.write_bytes(payload)
                with self.subTest(filename=filename), self.assertRaises(LeagueExportError):
                    load_export(path)

    def test_custom_player_with_no_pid_supported(self):
        raw = fixture()
        del raw["players"][0]["pid"]
        self.assertIsNone(normalize_export(raw).roster(0)[0].pid)


if __name__ == "__main__":
    unittest.main()
