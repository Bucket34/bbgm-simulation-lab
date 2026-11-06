# Experiment checkpoint (Steam Deck, November 5, 2026)

These measurements are **user-reported local tests**. They describe the toy environments, not actual Basketball GM gameplay.

- **LLM backend:** Alpaca-managed Ollama 0.40.14; Qwen3 8B Q4_K_M with Vulkan acceleration.
- **8B performance:** about 14.7–15 output tokens/s on short input at 8K context; 5,686-token raw-log input produced about 12.3 tokens/s. (The log test was a separate cybersecurity experiment.)
- **V1 memory replay:** 147 → 152 summed wins over the same three seeded 90-game toy scenarios; unseen memory/no-memory trial produced 142 vs. 142.
- **V2:** 400 training cases; 100 held-out; policy 89/100 optimal picks vs. OVR 64/100.
- **V4:** 500 new held-out cases; policy 436/500 optimal picks vs. OVR 318/500; +1.071 toy expected wins per scenario.
- **V4 disagreement stress test:** six deliberately selected disagreements; policy and hybrid 6/6 optimal vs. Qwen alone 0/6, but the selection is nonrepresentative.
- **V5:** 120 training episodes, 100 held-out, 3 seasons/episode. Learned policy +0.201 expected wins/season vs. greedy; better/worse/tied 19/29/52 episodes.
- **V5 Qwen held-out pilot:** 2 episodes; Qwen alone approx. 57.95 expected wins/season, Qwen + advice approx. 59.58; learned Python policy approx. 61.18. Too few episodes to generalize.
- **Next:** V5.1 controlled audit/guardrail retest. Validate whether policy advice and Qwen decisions diverge on matching team states, and record overrides separately from agent learning.

Important: Scripts with historic suffixes are retained for provenance. V5.1 builds on V5 and older policy experiments; it is not the actual BBGM engine.
