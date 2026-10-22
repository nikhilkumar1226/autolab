# Handoff notes (overnight autonomous run, 2026-10-22)

Written for Nikhil when you wake up. Updated after every phase. The detailed "what changed / why / how verified" notes for each phase are in `docs/CHANGELOG.md`; this file is the short version plus everything that needs you.

Rules I followed: provider `mock` (or `none`) only, no real LLM or paid API was ever called, no API key was read or printed, `.env` was never staged.

## Status

| Phase | Branch | Tag | Tests | Done-when |
|---|---|---|---|---|
| 2 (fix) | `phase-2-llm-agents` | `phase-2.1` | 230 passed | 2 of 4 pass; the other 2 need your keys |
| 3 | `phase-3-digital-twin` | `phase-3` | 311 passed | 2 of 2 pass |
| 4 | `phase-4-safety` | `phase-4` | 358 passed | 2 of 2 pass (red team: 59 cases, 0 unsafe commands reached the rig) |

## Decisions I made

- **Phase 2 tag.** `phase-2` already existed on GitHub (your save point at `55b8dc1`). I did not move it, because moving a pushed tag rewrites what others have fetched. The fixed Phase 2 is tagged **`phase-2.1`**.
- **Branches.** I kept your pattern of one branch per phase. Each new phase branch starts from the previous one, so the last branch contains everything. `main` is still at the initial commit; I did not merge or open pull requests (that is your review step).
- **Formula whitelist (Phase 2 fix).** I implemented exactly the whitelist you gave. Two small choices:
  - Unary plus (`+x`) is rejected, because the list names unary minus only.
  - Python builtin names (`open`, `eval`, `id`, …) cannot be parameter names. The evaluator never looks names up in Python, so this is defence in depth and readability, not a hole being closed.
  - sympy used to simplify the expression to detect "does not depend on x", "parameter has no effect" and `1/0`. Without sympy these checks are numerical, at fixed probe points. Parameter kinds (scale / location / offset) come from numerical derivatives. See CHANGELOG "Phase 2 fix" for the one known limitation.
- **sympy removed.** No code uses it now, so I removed it from `requirements.txt` and uninstalled it from `.venv` (`pip install sympy` puts it back if you ever need it).

### Phase 3 (details and numbers in CHANGELOG "Phase 3")
- **8 experiments, not 6.** The prompt lists 3 old + 5 new laws, so the "all 6 experiments" Done-when item was checked on all 8.
- **Stefan–Boltzmann temperature is in kK** (0.8–2.8 kK). In kelvin the power-law amplitude's relative SE was 22%, so the run could never converge.
- **No separate "saturating exponential" model.** It is the same family as the decaying exponential with a negative amplitude, and two identical models would tie on BIC forever. "Exponential decay" was renamed "Decaying exponential"; the key `exp_decay` is unchanged. A **growing exponential** was added for the diode.
- **New parsimony rule** (`analysis/parsimony.py`). Without it a linear law (Hooke) can never converge, because several 3-parameter models contain a line. It changes nothing for the Phase 1 experiments (checked: 0 of 60 sweep runs differ).
- **Outlier detection uses the best fit's prediction band, not the GP band the ROADMAP names.** The GP band was badly calibrated: 47 false alarms in 80 clean runs and 34 of 80 spikes missed, against 1 false alarm and 8 missed for the fit band. The GP band is still available as an option. I set the default before running that comparison.
- **Default fault schedule:** spike at cycle 6, stuck actuator at 8, dropout at 10–11. The ROADMAP example (stuck at 7, dropout at 9) cannot work as intended: cycle 7 is always the automatic re-measurement after the spike. You can still pass the ROADMAP's version as JSON with `--faults`.
- **Non-idealities are sized relative to each experiment's noise and position tolerance.** Drift was first sized relative to the response span. That was about 5 noise-σ per cycle for the precise diode and wrecked 19/20 runs, so I changed it to 0.3 noise-σ per cycle before the final sweep.
- **Faults count rig cycles:** one cycle is one reading or one pause, re-measurements included.

### Phase 4 (details in CHANGELOG "Phase 4", standards in docs/safety_and_standards.md)
- **Simulated rig clock.** The twin keeps simulated time (moves, readings, pauses), so the rate limit and watchdog are reproducible. Hardware uses wall time.
- **Defaults:** rate limit 5 moves/s; at most 3 moves in a row without a reading (my addition: the rate limit alone cannot stop a slow full-travel oscillation); watchdog 120 s, which is generous because real LLM calls sit between readings on hardware.
- **Step limits:** 10 N per step for the spring, 1 kK for the filament, 0.4 V for the diode. These are my judgement, not datasheet values; please check them. The GP now proposes only reachable points.
- **E-stop reset needs an operator name** (`reset_estop(operator="...")`), and it is logged.
- **The red team uses a scripted adversary** through a new `completion=` hook in `LLMClient`. It replaces only the network call and is not used anywhere else. No real model was called.
- **The twin now parks at the lowest safe position**, not at 0. A property test found that the LDR read 0 lux after homing and divided by zero.

## What you must do yourself

1. **`.env` layout.** Your `.env` still has the old Groq/Ollama layout. Copy the layout from `.env.example` (Claude first, then the two Gemini models) and add `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. I did not touch `.env`.
2. **Phase 2 open Done-when items** (need keys):
   - `python -m llm.check` → should print OK for Claude and Gemini.
   - `python main.py --provider litellm` → check the run JSON in `runs/` shows the real prompts, responses, tokens and cost.

3. **Red team against a real model** (optional, costs a little): `python -m safety.redteam --provider litellm --out results/redteam_litellm.csv` runs the prompt-injection cases against Claude/Gemini.

## Open questions

- **Random outliers are rarely caught** (8 of 68 in the sweep), because most runs stop at n = 8, when a fit has too few degrees of freedom to call anything an outlier at p = 0.001. Option: calibrate each instrument's noise once (repeated readings at one position) and use it for detection. Should we add that calibration step?
- **Hooke with faults converged on cos² in 4/20 runs** (cos² over 0–20 "degrees" is almost a straight line). Raising `min_points` for all experiments, or turning on predictive stability, would probably fix it but costs measurements. Your call after the Phase 5 benchmark.
