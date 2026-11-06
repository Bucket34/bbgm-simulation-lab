# GM core (incremental V5.1 refactor)

This directory is the first step toward modularizing the BBGM AI Agent Lab.
It does **not** connect to the actual Basketball GM engine, retrain Qwen3,
or change the frozen V5 simulator.

- `policy.py`: chooses a legal signing using the existing V5 learned policy.
- `controller.py`: pure validation/override logic (no Ollama calls).
- `rationale.py`: simple text warnings for two known explanation mistakes.

The existing `bbgm_deck_guardrails_v5_1.py` remains the command-line entry
point, imports these components, and retains its previous JSON report format.
The existing V5 simulator still owns roster changes, salary-cap accounting,
score calculations, and Ollama communication; subsequent refactors can
extract those independently.

From the repository root:

```bash
python3 -m unittest discover -s agents/basketball-gm/tests -v
```

The eight original V5.1 regression tests must keep passing. New unit tests
also exercise the independent controller without starting Ollama.
