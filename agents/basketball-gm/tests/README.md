# Offline regression tests — BBGM Deck V5.1

Run from the repository root:

```bash
python3 -m unittest discover -s agents/basketball-gm/tests -v
```

Checks eight controller behaviors, including legal choices, known Episode 2 recovery,
invalid Qwen recommendations, override thresholds, and agreement with V5's
learned-policy baseline. Uses deterministic mock decisions instead of Ollama.

These are **synthetic-simulator** regression tests, not a validation of actual
Basketball GM gameplay. They require only the Python standard library, contact
no network endpoints, and do not write model or simulation result files.
