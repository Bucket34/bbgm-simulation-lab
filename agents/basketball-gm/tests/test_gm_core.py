"""Focused pure-controller tests for the extracted V5.1 decision module."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm_core import resolve_decision

LEGAL = {'Y1-A', 'Y1-B'}
SCORES = {'Y1-A': 8.116, 'Y1-B': 0.2}


class ControllerUnitTests(unittest.TestCase):
    def test_unguarded_modes_preserve_legal_proposals(self):
        for mode in ('alone', 'advised'):
            with self.subTest(mode=mode):
                result = resolve_decision({'choice': 'Y1-B'}, LEGAL, 'Y1-A', SCORES, mode, 1.0)
                self.assertEqual(result.final_action, 'Y1-B')
                self.assertEqual(result.override_reason, '')

    def test_guarded_overrides_when_policy_gap_exceeds_margin(self):
        result = resolve_decision({'choice': 'PASS'}, LEGAL, 'Y1-A', SCORES, 'guarded', 1.0)
        self.assertEqual(result.final_action, 'Y1-A')
        self.assertEqual(result.estimated_gain, 8.116)
        self.assertIn('threshold 1.000', result.override_reason)

    def test_guarded_keeps_proposal_below_margin(self):
        result = resolve_decision({'choice': 'Y1-B'}, LEGAL, 'Y1-A', SCORES, 'guarded', 9.0)
        self.assertEqual(result.final_action, 'Y1-B')
        self.assertEqual(result.override_reason, '')

    def test_invalid_fallback_changes_only_guarded(self):
        for mode, expected in [('alone', 'PASS'), ('advised', 'PASS'), ('guarded', 'Y1-A')]:
            with self.subTest(mode=mode):
                result = resolve_decision({'choice': 'Y1-X'}, LEGAL, 'Y1-A', SCORES, mode, 1.0)
                self.assertTrue(result.invalid)
                self.assertEqual(result.final_action, expected)

    def test_does_not_accept_illegal_policy_recommendation(self):
        with self.assertRaises(ValueError):
            resolve_decision({'choice': 'PASS'}, LEGAL, 'Y1-X', SCORES, 'guarded', 1.0)

    def test_rejects_unknown_mode_and_negative_margin(self):
        with self.assertRaises(ValueError):
            resolve_decision({'choice': 'PASS'}, LEGAL, 'PASS', SCORES, 'other', 1.0)
        with self.assertRaises(ValueError):
            resolve_decision({'choice': 'PASS'}, LEGAL, 'PASS', SCORES, 'guarded', -1.0)


if __name__ == '__main__':
    unittest.main()
