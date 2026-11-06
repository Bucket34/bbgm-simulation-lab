"""Phase 3: compare the extracted simulator with frozen original V5.

These tests run entirely offline and do not modify any saved experiments.
"""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import bbgm_deck_multiseason_v5 as frozen
import bbgm_deck_guardrails_v5_1 as guard
from gm_core import simulator as extracted
from gm_core import policy as core_policy
from gm_core import llm as core_llm

PARAMS = [(0.7, 1.5, 1.5), (0.0, 0.0, 0.0), (1.3, 3.0, 0.75)]
TEST_SEED = 37001 + 100000003


class SimulatorCompatibilityTests(unittest.TestCase):
    def test_active_modules_use_extracted_environment(self):
        self.assertIs(guard.sim, extracted)
        self.assertIs(core_policy.sim, extracted)
        self.assertIs(core_llm.available_offers, extracted.available_offers)
        self.assertIsNot(extracted, frozen)

    def test_episode_generation_matches_frozen_v5(self):
        for seed in (37001, TEST_SEED, 26001, 17001):
            for idx in (0, 1, 2, 9, 41):
                for years in (2, 3, 5):
                    with self.subTest(seed=seed, idx=idx, years=years):
                        a = extracted.make_episode(seed, idx, years)
                        b = frozen.make_episode(seed, idx, years)
                        self.assertEqual(a, b)

    def test_scoring_aging_offers_and_policy_match(self):
        for idx in range(25):
            episode = frozen.make_episode(TEST_SEED, idx, 3)
            signings = []
            for year in range(3):
                with self.subTest(idx=idx, year=year):
                    self.assertEqual(extracted.available_offers(episode, signings, year),
                                     frozen.available_offers(episode, signings, year))
                    active = episode['initial'] + [p for p in signings if p['ends'] > year]
                    self.assertEqual(extracted.strength(active, year), frozen.strength(active, year))
                    self.assertEqual(extracted.expected_wins(active, year), frozen.expected_wins(active, year))
                    self.assertEqual(extracted.choose_greedy(episode, signings, year),
                                     frozen.choose_greedy(episode, signings, year))
                    for params in PARAMS:
                        self.assertEqual(extracted.learned_estimates(episode, signings, year, params),
                                         frozen.learned_estimates(episode, signings, year, params))
                        self.assertEqual(extracted.choose_learned(episode, signings, year, params),
                                         frozen.choose_learned(episode, signings, year, params))
                    legal = frozen.available_offers(episode, signings, year)
                    for offer in legal:
                        self.assertEqual(extracted.candidate_gains(episode, signings, year, offer),
                                         frozen.candidate_gains(episode, signings, year, offer))
                    if legal:
                        signings.append(legal[0])

    def test_full_season_history_exactly_matches_frozen(self):
        for idx in range(35):
            episode = frozen.make_episode(TEST_SEED, idx, 3)
            for label, old, new in (
                ('pass', lambda e,s,y: None, lambda e,s,y: None),
                ('greedy', frozen.choose_greedy, extracted.choose_greedy),
                ('learned', lambda e,s,y: frozen.choose_learned(e,s,y,PARAMS[0]),
                            lambda e,s,y: extracted.choose_learned(e,s,y,PARAMS[0])),
                ('first-legal', lambda e,s,y: next(iter(frozen.available_offers(e,s,y)),None),
                                lambda e,s,y: next(iter(extracted.available_offers(e,s,y)),None)),
            ):
                with self.subTest(idx=idx, strategy=label):
                    original = frozen.run_episode(episode, old)
                    migrated = extracted.run_episode(episode, new)
                    self.assertEqual(migrated, original)

    def test_prompt_contract_still_matches_frozen_v5(self):
        for idx in range(8):
            episode = frozen.make_episode(TEST_SEED, idx, 3)
            for year in range(3):
                # Same legal offer data and current cap before any signing.
                prompt, legal = core_llm.build_choice_prompt(episode, [], [], year)
                self.assertEqual(legal, [p['id'] for p in frozen.available_offers(episode,[],year)])
                self.assertEqual(prompt['legal_offers'], frozen.available_offers(episode,[],year))
                future = {p['id'] for y in range(year+1, episode['years'])
                          for p in episode['offers_by_year'][y]}
                self.assertTrue(future.isdisjoint(set(legal)))

    def test_known_episode_two_controller_outcome_unchanged(self):
        episode = frozen.make_episode(TEST_SEED, 1, 3)
        def pass_ai(*args, **kwargs):
            return {'choice': 'PASS', 'reason': 'Testing deterministic behavior'}
        result = guard.play(episode, PARAMS[0], 'guarded', 1.0, 'unused', 'unused', ai_choice=pass_ai)
        self.assertEqual([season['final_action'] for season in result['seasons']],
                         ['PASS', 'Y1-A', 'PASS'])
        baseline_total, _ = frozen.run_episode(
            episode, lambda e,s,y: frozen.choose_learned(e,s,y,PARAMS[0]))
        self.assertEqual(result['total_wins'], baseline_total)


if __name__ == '__main__':
    unittest.main()
