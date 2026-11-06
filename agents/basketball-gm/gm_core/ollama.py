"""Minimal local Ollama HTTP client used by the synthetic GM lab.

Only JSON is exchanged with Ollama. No LLM-generated shell commands are run.
The V5 historical experiment retains its original client implementation.
"""
from __future__ import annotations

import json
import urllib.request


def api(base, path, payload=None, timeout=240):
    """Call an Ollama endpoint and decode its JSON response."""
    raw = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base.rstrip('/') + path, data=raw,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_gpu(base, model):
    """Verify that Ollama reports the model's entire size allocated to VRAM.

    This checks Ollama's reported allocation, not actual inference performance.
    It deliberately retains the original V5 fail-closed behavior.
    """
    version = api(base, '/api/version', timeout=10).get('version')
    api(base, '/api/generate', {"model": model, "stream": False,
            "keep_alive": "5m", "options": {"num_ctx": 8192}})
    models = api(base, '/api/ps', timeout=10).get('models', [])
    loaded = next((m for m in models if m.get('name') == model), None)
    if not loaded or not loaded.get('size') or loaded['size'] != loaded['size_vram']:
        raise RuntimeError('Full Vulkan GPU offload not confirmed; skipping LLM experiment')
    print(f'Ollama {version}, GPU allocation: {loaded["size_vram"]:,}/{loaded["size"]:,}', flush=True)


def unload_model(base, model):
    """Ask Ollama to unload the model (the caller handles failures)."""
    return api(base, '/api/generate', {'model': model,
                                     'stream': False, 'keep_alive': 0}, timeout=45)
