# GM core (incremental V5.1 refactor)

The `gm_core` modules are an incremental extraction from V5.1. They preserve
its **synthetic** basketball engine and do not connect to the real Basketball GM
engine, alter league saves, or retrain Qwen3.

| Module | Responsibility |
|---|---|
| `simulator.py` | V5-compatible, deterministic toy basketball environment: seasons, contracts, scoring, legal signings |
| `policy.py` | Choose a legal signing using the existing V5 learned policy |
| `controller.py` | Pure legal-action validation and policy-based overrides |
| `rationale.py` | Narrow, deterministic warnings about misleading explanations |
| `llm.py` | Build synthetic GM prompts, validate model JSON, propose signings |
| `ollama.py` | Local Ollama JSON API, GPU-allocation check, model unload |

`bbgm_deck_guardrails_v5_1.py` remains the V5.1 command-line entry point and
now uses these modules. The historical `bbgm_deck_multiseason_v5.py` remains
**unchanged** so it can serve as a compatibility reference. The active V5.1
command now imports `gm_core.simulator` rather than importing its engine
from the historical V5 experiment.

The extracted `llm_choice` has the same function arguments, JSON request, and
response shape as the original V5 implementation. It does **not** execute any
text as code. Rejected responses become `PASS` with `invalid=True`, which the
controller may handle according to its mode. No decision is a real league action.

The LLM *prompt* still depends on the toy basketball schema. The transport is
independent, which will make additional agents and a future real BBGM adapter
simpler to add without changing the controller.

Run tests from the repository root (no Alpaca required):

```bash
python3 -m unittest discover -s agents/basketball-gm/tests -v
```

Phase 2 adds offline contract tests that compare outgoing requests and parsed
responses with the original V5 implementation. The Phase 2 live Ollama smoke test passed on the Steam Deck before this
simulator extraction. Re-run a short live test after Phase 3 installs.

## Phase 3 — Simulator extraction

`simulator.py` holds the original V5 toy world: roster generation, invented
player statistics, deterministic aging and expected-wins math, cap/contract
logic, legal offers, scoring, and season transitions. The original V5 script
continues to exist unchanged as an independent regression oracle.

The controller uses the simulator through this module, but the *interface* is
still toy-specific. We have **not** connected an actual Basketball GM engine or
created a generalized interface yet. Future work is a real BBGM data adapter
with separate validation and explicit, permission-controlled writes.

The new offline tests compare core results against the frozen V5 source for
multiple episode seeds, policies, and seasons; no model or internet needed.
