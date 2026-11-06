"""Offline contract tests for modular Ollama/LLM integration in V5.1.

No network calls, model inference, reports, or Qwen weights are involved.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import bbgm_deck_multiseason_v5 as sim
import bbgm_deck_guardrails_v5_1 as guard
from gm_core import llm as core_llm
from gm_core import ollama

EPISODE = sim.make_episode(37001 + 100000003, 1, 3)


class LLMContractTests(unittest.TestCase):
    def test_prompt_exposes_only_current_legal_offers(self):
        prompt, ids = core_llm.build_choice_prompt(EPISODE, [], [], 0)
        expected = [p['id'] for p in sim.available_offers(EPISODE, [], 0)]
        self.assertEqual(ids, expected)
        self.assertEqual(prompt['legal_candidate_ids'], expected + ['PASS'])
        self.assertEqual({p['id'] for p in prompt['legal_offers']}, set(expected))
        self.assertEqual(prompt['available_cap'], EPISODE['cap'])
        self.assertNotIn('Y1-', json.dumps(prompt))
        self.assertNotIn('Y2-', json.dumps(prompt))

    def test_advisory_scores_opt_in_and_rounded(self):
        plain, _ = core_llm.build_choice_prompt(EPISODE, [], [], 0)
        advice, _ = core_llm.build_choice_prompt(EPISODE, [], [], 0,
                                                {'Y0-A': 2.1234567})
        self.assertNotIn('python_policy', plain)
        self.assertEqual(advice['python_policy']['scores'], {'Y0-A': 2.123})

    def test_parses_legal_choice_and_speed(self):
        value = core_llm.parse_choice_response({
            'done_reason': 'stop',
            'response': json.dumps({'choice': ' y1-a ', 'reason': 'x' * 250}),
            'eval_duration': 1000000000, 'eval_count': 15,
        }, ['Y1-A'])
        self.assertEqual(value['choice'], 'Y1-A')
        self.assertEqual(len(value['reason']), 220)
        self.assertEqual(value['speed'], 15)
        self.assertFalse(value['invalid'])

    def test_rejects_illegal_choice(self):
        for choice in ('Y1-B', 'UNKNOWN', ''):
            with self.subTest(choice=choice):
                value = core_llm.parse_choice_response({
                    'done_reason': 'stop',
                    'response': json.dumps({'choice': choice, 'reason': 'Looks useful'}),
                }, ['Y1-A'])
                self.assertEqual(value['choice'], 'PASS')
                self.assertTrue(value['invalid'])

    def test_rejects_truncated_and_malformed_json(self):
        for raw in ({'done_reason': 'length', 'response': '{'},
                    {'done_reason': 'stop', 'response': '{bad'}):
            with self.subTest(raw=raw):
                value = core_llm.parse_choice_response(raw, ['Y1-A'])
                self.assertTrue(value['invalid'])
                self.assertEqual(value['choice'], 'PASS')

    def test_matches_legacy_v5_request_and_result(self):
        """Compare all request bytes and returned fields with frozen V5 code."""
        legal = sim.available_offers(EPISODE, [], 0)
        options = [p['id'] for p in legal] + ['PASS', 'ILLEGAL']
        for advice in (None, sim.learned_estimates(EPISODE, [], 0, (0.7, 1.5, 1.5))):
            for choice in options:
                with self.subTest(advice=advice is not None, choice=choice):
                    response = {
                        'done_reason': 'stop',
                        'response': json.dumps({'choice': choice, 'reason': 'testing'}),
                        'eval_duration': 2000000000, 'eval_count': 30,
                    }
                    captured_legacy = []
                    captured_new = []

                    def fake_factory(captured):
                        def fake_api(base, path, payload=None, timeout=240):
                            captured.append((base, path, payload, timeout))
                            return response
                        return fake_api

                    args = (EPISODE, [], [], 0, 'qwen3:8b', 'http://127.0.0.1:11435', advice)
                    with patch.object(sim, 'api', side_effect=fake_factory(captured_legacy)):
                        old = sim.llm_choice(*args)
                    with patch.object(core_llm, 'api', side_effect=fake_factory(captured_new)):
                        new = core_llm.llm_choice(*args)
                    self.assertEqual(captured_new, captured_legacy)
                    self.assertEqual(new, old)

    def test_guard_main_loop_uses_extracted_client(self):
        calls = []
        def fake_api(base, path, payload=None, timeout=240):
            calls.append((base, path, payload))
            return {'done_reason': 'stop', 'response': '{"choice":"PASS", "reason":"OK"}'}
        with patch.object(core_llm, 'api', side_effect=fake_api):
            result = guard.play(EPISODE, (0.7, 1.5, 1.5), 'alone', 1.0,
                                'qwen3:8b', 'http://127.0.0.1:11435')
        self.assertEqual(len(calls), 3)
        self.assertEqual([s['final_action'] for s in result['seasons']], ['PASS'] * 3)
        self.assertEqual(result['invalid_proposals'], 0)


class OllamaClientTests(unittest.TestCase):
    def test_api_posts_json(self):
        def fake_urlopen(request, timeout):
            self.assertEqual(request.full_url, 'http://127.0.0.1:11435/api/generate')
            self.assertEqual(json.loads(request.data), {'model': 'qwen3:8b'})
            self.assertEqual(timeout, 5)
            return io.BytesIO(b'{"ok":true}')
        with patch.object(ollama.urllib.request, 'urlopen', side_effect=fake_urlopen):
            result = ollama.api('http://127.0.0.1:11435/', '/api/generate',
                                {'model': 'qwen3:8b'}, timeout=5)
        self.assertEqual(result, {'ok': True})

    def test_verify_gpu_accepts_full_allocation(self):
        def fake_api(base, path, payload=None, timeout=240):
            if path == '/api/version':
                return {'version': '0.40.14'}
            if path == '/api/ps':
                return {'models': [{'name': 'qwen3:8b', 'size': 100, 'size_vram': 100}]}
            return {}
        out = io.StringIO()
        with patch.object(ollama, 'api', side_effect=fake_api), contextlib.redirect_stdout(out):
            ollama.verify_gpu('http://127.0.0.1:11435', 'qwen3:8b')
        self.assertIn('GPU allocation: 100/100', out.getvalue())

    def test_verify_gpu_rejects_partial_allocation(self):
        def fake_api(base, path, payload=None, timeout=240):
            if path == '/api/version':
                return {'version': '0.40.14'}
            if path == '/api/ps':
                return {'models': [{'name': 'qwen3:8b', 'size': 100, 'size_vram': 50}]}
            return {}
        with patch.object(ollama, 'api', side_effect=fake_api):
            with self.assertRaisesRegex(RuntimeError, 'offload not confirmed'):
                ollama.verify_gpu('http://127.0.0.1:11435', 'qwen3:8b')

    def test_unload_sends_keep_alive_zero(self):
        with patch.object(ollama, 'api', return_value={'done': True}) as send:
            ollama.unload_model('http://127.0.0.1:11435', 'qwen3:8b')
        send.assert_called_once_with('http://127.0.0.1:11435', '/api/generate',
                                     {'model': 'qwen3:8b', 'stream': False,
                                      'keep_alive': 0}, timeout=45)


if __name__ == '__main__':
    unittest.main()
