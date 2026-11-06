"""Offline regression tests for the V5.1 synthetic GM controller.

Run from the repository root:
    python3 -m unittest discover -s agents/basketball-gm/tests -v

These tests do not contact Ollama or change any report/model/league files.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# The existing project uses hyphens in its folder name and imports sibling
# modules directly. Make those modules available without changing production code.
AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import bbgm_deck_multiseason_v5 as sim
import bbgm_deck_guardrails_v5_1 as guard

PARAMS = (0.7, 1.5, 1.5)
TEST_SEED = 37001 + 100000003


def episode_two():
    # Exactly the episode used in the V5/V5.1 experiments (1-based #2).
    return sim.make_episode(TEST_SEED, 1, 3)


def always_pass(*args, **kwargs):
    return {'choice': 'PASS', 'reason': 'Declining all offers'}


def policy_as_ai(episode, signings, history, year, model, base, advice):
    choice, _ = guard.policy_pick(episode, signings, year, PARAMS)
    return {'choice': choice, 'reason': 'Uses policy recommendation'}


class DecisionGuardrailTests(unittest.TestCase):
    def test_episode_generation_is_repeatable(self):
        self.assertEqual(episode_two(), episode_two())

    def test_policy_recommendations_are_legal(self):
        for idx in range(20):
            episode = sim.make_episode(TEST_SEED, idx, 3)
            signings = []
            for year in range(episode['years']):
                recommended, scores = guard.policy_pick(episode, signings, year, PARAMS)
                offers = {p['id']: p for p in sim.available_offers(episode, signings, year)}
                self.assertTrue(set(scores).issubset(offers))
                self.assertIn(recommended, set(offers) | {'PASS'})
                if recommended != 'PASS':
                    signings.append(offers[recommended])

    def test_unguarded_always_passes_when_requested(self):
        ep = episode_two()
        result = guard.play(ep, PARAMS, 'alone', 1.0, 'unused', 'unused', ai_choice=always_pass)
        self.assertEqual([s['final_action'] for s in result['seasons']], ['PASS'] * 3)
        self.assertEqual(result['overrides'], 0)
        self.assertEqual(result['invalid_proposals'], 0)
        expected_total, _ = sim.run_episode(ep, lambda e, s, y: None)
        self.assertAlmostEqual(result['total_wins'], expected_total, places=8)

    def test_guarded_repairs_known_episode_two_missed_signing(self):
        ep = episode_two()
        result = guard.play(ep, PARAMS, 'guarded', 1.0, 'unused', 'unused', ai_choice=always_pass)
        self.assertEqual([s['final_action'] for s in result['seasons']], ['PASS', 'Y1-A', 'PASS'])
        self.assertEqual(result['overrides'], 1)
        self.assertEqual(result['invalid_proposals'], 0)
        learned_total, _ = sim.run_episode(ep, lambda e, s, y: sim.choose_learned(e, s, y, PARAMS))
        self.assertAlmostEqual(result['total_wins'], learned_total, places=8)

    def test_invalid_ai_choice_never_becomes_an_illegal_signing(self):
        def invalid(*args, **kwargs):
            return {'choice': 'OVER_BUDGET_OR_UNKNOWN', 'reason': 'mistaken ID', 'invalid': True}

        ep = episode_two()
        for mode in ('alone', 'advised', 'guarded'):
            with self.subTest(mode=mode):
                res = guard.play(ep, PARAMS, mode, 1.0, 'unused', 'unused', ai_choice=invalid)
                self.assertEqual(res['invalid_proposals'], 3)
                for season in res['seasons']:
                    self.assertIn(season['final_action'], season['legal_choices'])
                    self.assertTrue(season['qwen_invalid'])
                if mode != 'guarded':
                    self.assertEqual([s['final_action'] for s in res['seasons']], ['PASS'] * 3)

    def test_high_margin_preserves_legal_qwen_passes(self):
        result = guard.play(episode_two(), PARAMS, 'guarded', 50.0, 'unused', 'unused', ai_choice=always_pass)
        self.assertEqual(result['overrides'], 0)
        self.assertEqual([s['final_action'] for s in result['seasons']], ['PASS'] * 3)

    def test_accepting_learned_policy_matches_baseline(self):
        ep = episode_two()
        result = guard.play(ep, PARAMS, 'guarded', 1.0, 'unused', 'unused', ai_choice=policy_as_ai)
        baseline_total, baseline_history = sim.run_episode(
            ep, lambda e, s, y: sim.choose_learned(e, s, y, PARAMS))
        self.assertAlmostEqual(result['total_wins'], baseline_total, places=8)
        self.assertEqual([s['final_action'] for s in result['seasons']],
                         [s['choice'] for s in baseline_history])
        self.assertEqual(result['overrides'], 0)

    def test_rationale_checks_are_narrow_and_deterministic(self):
        flags = guard.rationale_flags('Salary cap is exhausted.', {'Y1-A': 8.116}, cap=14)
        self.assertTrue(any('cap is exhausted' in flag for flag in flags))
        flags = guard.rationale_flags('All offers have negative value.', {'Y1-A': 8.116}, cap=14)
        self.assertTrue(any('all offers' in flag.lower() for flag in flags))
        self.assertEqual(guard.rationale_flags('The cap is exhausted.', {}, cap=0), [])


if __name__ == '__main__':
    unittest.main()
