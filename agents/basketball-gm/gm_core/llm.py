"""Synthetic BBGM free-agent prompts and structured Ollama responses.

The public llm_choice signature and request format match V5. This keeps the
simulator and controller ignorant of the HTTP transport implementation while
preserving the legacy experiment's observable behavior.

The prompt is specific to the current toy basketball simulation. A future real
BBGM adapter can reuse the Ollama transport and response validation separately.
"""
from __future__ import annotations

import json

from .simulator import available_offers
from .ollama import api


def build_choice_prompt(episode, signings, history, year, advice=None):
    """Return (prompt object, legal IDs) using only currently visible offers."""
    legal = available_offers(episode, signings, year)
    legal_ids = [p['id'] for p in legal]
    cap_left = episode['cap'] - sum(p['cost'] for p in signings if p['ends'] > year)
    active_roster = episode['initial'] + [p for p in signings if p['ends'] > year]
    prompt_obj = {
        "role": "Fictional basketball general manager; choose ONE legal free-agent signing "
                "or PASS to maximize expected wins across all remaining seasons. "
                "Contracts block cap space until expiry. Young players develop; older players decline. "
                "Do not assume future free agents are known. Reply with ONLY JSON "
                '{"choice":"Y0-A" or "PASS", "reason":"short explanation"}.',
        "season": year+1, "seasons_remaining": episode['years'] - year,
        "salary_cap": episode['cap'], "available_cap": cap_left,
        "legal_candidate_ids": legal_ids + ['PASS'],
        "signed_roster": active_roster,
        "legal_offers": legal,
        "previous_seasons": history,
    }
    if advice is not None:
        prompt_obj['python_policy'] = {"description": "Estimated long-term value, not confirmed outcomes. "
                                          "May recommend PASS. You may disagree.",
                                        "scores": {k: round(v, 3) for k, v in advice.items()}}
    return prompt_obj, legal_ids


def parse_choice_response(response, legal_ids):
    """Normalize a response; reject malformed/illegal decisions as PASS."""
    result = {"choice": "PASS", "reason": "", "invalid": False,
              "finish": response.get('done_reason')}
    try:
        if response.get('done_reason') == 'length':
            raise ValueError('Truncated output')
        data = json.loads(response.get('response', ''))
        ident = str(data.get('choice', '')).strip().upper()
        if ident == 'PASS':
            result['choice'] = ident
        elif ident in legal_ids:
            result['choice'] = ident
        else:
            raise ValueError(f'Illegal choice: {ident}')
        result['reason'] = str(data.get('reason', ''))[:220]
    except (ValueError, TypeError, AttributeError) as e:
        result['invalid'] = True
        result['reason'] = f'Rejected decision, default PASS: {e}'
    duration = response.get('eval_duration') or 0
    result['speed'] = round(response.get('eval_count', 0) * 1e9 / duration, 2) if duration else None
    return result


def llm_choice(episode, signings, history, year, model, base, advice=None):
    """Ask a local Ollama model for one *proposed* free-agent action."""
    prompt_obj, legal_ids = build_choice_prompt(episode, signings, history, year, advice)
    response = api(base, '/api/generate', {
        "model": model, "prompt": json.dumps(prompt_obj, separators=(',', ':')),
        "stream": False, "think": False, "format": "json", "keep_alive": "5m",
        "options": {"num_ctx": 8192, "num_predict": 220, "temperature": 0, "seed": 42},
    })
    return parse_choice_response(response, legal_ids)
