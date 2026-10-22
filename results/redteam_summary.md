# Red-team summary

Provider: `scripted`. Every case ran through the real pipeline with a spy rig recording every command that reached the hardware. Per-case results: `redteam.csv`.

| Attempted | Blocked at LLM/agent level | Blocked at the guard | Harmless (inside the envelope) | No effect | **Unsafe commands reaching the rig** |
|---|---|---|---|---|---|
| 59 | 23 | 29 | 3 | 4 | **0** |

| Category | Outcomes |
|---|---|
| code_injection | blocked_llm 5 |
| direct_call | blocked_guard 4, harmless 1 |
| estop_watchdog | blocked_guard 3 |
| malformed_json | blocked_guard 1, blocked_llm 7 |
| numeric_edge | blocked_guard 2, blocked_llm 5 |
| oscillation_rate | blocked_guard 3, harmless 1 |
| out_of_range | blocked_guard 9, blocked_llm 1 |
| prompt_injection | blocked_guard 4, no_effect 4 |
| unit_trick | blocked_guard 3, blocked_llm 5, harmless 1 |

Skipped (need the scripted adversary): 0.
Outcomes different from the expectation in attacks.yaml: none.
