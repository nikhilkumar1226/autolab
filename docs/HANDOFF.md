# Handoff notes (overnight autonomous run, 2026-10-22)

Written for Nikhil. The detailed "what changed / why / how it was verified" for every phase is in `docs/CHANGELOG.md`; this file is the short version plus everything that needs you.

**Summary:**
- **Done:** Phase 2 (with the AST formula fix), then Phases 3 to 7, each with its checkpoint: tests green, CHANGELOG entry, `Phase N: …` commit, tag, push, ROADMAP boxes.
- **Tests:** 404 pass offline in about 2 minutes (230 at the start).
- **Not done:** everything that needs your API keys, the Wokwi licence, or a browser. Those are listed below.

**What I never did:**
- call a real LLM or any paid API (provider `mock` or `none` throughout);
- read or print a key (the `.env` keys were empty anyway), or stage `.env`;
- merge anything into `main`.

## Status

| Phase | Branch | Tag | Tests | Done-when |
|---|---|---|---|---|
| 2 (fix) | `phase-2-llm-agents` | `phase-2.1` | 230 | 2 of 4 pass; 2 need your keys |
| 3 Digital twin, experiments, faults | `phase-3-digital-twin` | `phase-3` | 311 | 2 of 2 |
| 4 Safety hardening, red team | `phase-4-safety` | `phase-4` | 358 | 2 of 2 (59 attacks, 0 unsafe commands reached the rig) |
| 5 Benchmark | `phase-5-benchmark` | `phase-5` | 367 | 1 of 3 (`--quick` in 142 s); the real-LLM run is yours |
| 6 Firmware, Wokwi | `phase-6-firmware` | `phase-6` | 391 | 0 of 2, both need Wokwi; firmware compiles, and a full run passes against a fake port |
| 7 Dashboard | `phase-7-dashboard` | `phase-7` | 404 | 1 of 1 verified headless; please also look in a browser |

Each branch starts from the previous one, so `phase-7-dashboard` contains everything. `main` is still at the initial commit. When you have reviewed it, merging `phase-7-dashboard` into `main` is a fast-forward.

## Try it

```bash
source .venv/bin/activate
pytest -q                                             # 404 passed
streamlit run app/Home.py                             # dashboard: Live Lab, Run History, Benchmark, System
python main.py --experiment diode_iv --faults default --nonideal --provider mock
python -m safety.redteam                              # 59 attacks -> results/redteam.csv, 0 unsafe
python -m benchmark.run --quick                       # about 2 minutes -> results/quick/
```

Results for the report are in `results/`: `RESULTS.md`, `benchmark_*.csv`, `figures/`, `redteam*.{csv,md}`, `phase3_twin_sweep.csv` and the Phase 1 sweeps. **The LLM-method rows in the benchmark are mock.** RESULTS.md says so at the top.

## What you must do yourself

1. **Update the `.env` layout.** Your `.env` still has the old Groq/Ollama layout. Copy the layout from `.env.example` (Claude first, then the two Gemini models) and add `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. I did not touch `.env`.
2. **Phase 2 checks** (need keys):
   - `python -m llm.check` should print OK for Claude and Gemini.
   - `python main.py --provider litellm`: then check that the run JSON in `runs/` shows the real prompts, answers, tokens and cost.
3. **The real-LLM benchmark** (Phase 5 Done-when), with the Claude key in `.env`:
   - Look first: `python -m benchmark.run --full --provider litellm --dry-run`.
   - Small first run (2 seeds, clean twin, about 1,600 calls): `python -m benchmark.run --full --provider litellm --llm-seeds 2 --conditions clean --workers 2`.
   - Full run: `python -m benchmark.run --full --provider litellm --workers 2`. The mock policies made about 6,700 calls. That is estimated at about $0.6 at litellm's bundled Haiku price, but check the Anthropic Usage page after the first runs.
   - The Gemini free tier (about 20 per day on Flash) cannot cover this, so Claude credit is needed. Answers are cached, so a rerun is free. The baseline rows are reused; only the LLM methods rerun.
   - Then send me `results/`.
4. **Red team against a real model** (optional, cheap): `python -m safety.redteam --provider litellm --out results/redteam_litellm.csv`. It runs the prompt-injection cases only.
5. **Wokwi** (Phase 6 Done-when):
   1. Install the Wokwi Simulator extension in VS Code.
   2. F1, then **Wokwi: Request a New License**.
   3. F1, then **Wokwi: Start Simulator**, and keep its tab visible.
   4. Run `python main.py --experiment malus_law --hardware --port rfc2217://localhost:4000 --provider mock`.
   5. Screenshot the circuit and save it as `docs/img/wokwi_circuit.png`; the System page shows it.

   Details are in `docs/wokwi.md`.
6. **Dashboard in a browser** (Phase 7): run `streamlit run app/Home.py`, start a run on Live Lab, and press E-STOP once mid-run. I checked this headless only.
   - To use the real LLM there, also set `APP_PASSWORD` in `.env` and type it into the sidebar.
   - Without a password the dashboard always uses mock, even locally. That is deliberate, to protect the credit on the hosted app.
7. **Review my judgement calls** (listed below), in particular the step limits and the default fault schedule.

## Decisions I made

### Phase 2 fix
- **Phase 2 tag.** `phase-2` already existed on GitHub. I did not move it, because moving a pushed tag rewrites history others may have fetched. The fixed version is **`phase-2.1`**.
- **The formula whitelist is exactly yours.** Two small additions:
  - unary plus is rejected;
  - Python builtin names (`open`, `eval`, …) cannot be parameter names.
- **sympy's checks became numerical.** sympy simplification used to catch "no x", "parameter has no effect" and `1/0`. These are now numerical checks at fixed probe points, and parameter kinds come from numerical derivatives.
- **sympy removed** from `requirements.txt` and from `.venv`.

### Phase 3
- **8 experiments.** "All 6 experiments" was checked on all 8 (3 old + 5 new).
- **Stefan–Boltzmann temperature is in kK.** In kelvin, the power-law amplitude's relative SE was 22% and the run could never converge.
- **No separate saturating exponential.** It is the decaying exponential with a negative amplitude, and duplicate models tie on BIC forever. That model is renamed "Decaying exponential" (key unchanged), and a growing exponential was added for the diode.
- **Parsimony rule.** A model with more parameters that has the same curve as a simpler one is not counted as a rival. Without this, a linear law can never converge. It changes none of the Phase 1 results (0 of 60 runs differ).
- **Outlier detection uses the best fit's prediction band (Student-t), not the GP band** the ROADMAP names. The GP band gave 47 false alarms in 80 clean runs; the fit band gave 1. I set the default before running the comparison.
- **Default fault schedule:** spike at cycle 6, stuck actuator at 8, dropout at 10–11. In the ROADMAP's example, the stuck actuator at cycle 7 always falls on the spike's re-measurement.
- **Non-idealities scale with each experiment's noise and position tolerance.** Drift is 0.3 noise-σ per cycle; the first version, relative to the response span, swamped the precise diode.

### Phase 4
- **The twin keeps a simulated rig clock**, so the rate limit and watchdog are reproducible.
- **Moves-without-reading interlock (3)**, my addition: a slow full-travel oscillation stays under any sensible rate limit.
- **Rate limit 10 moves/s** (raised from 5 in Phase 6, because the real firmware can legitimately reach about 8/s).
- **Watchdog 120 s of rig time**, generous because real LLM calls happen between readings on hardware.
- **Step limits: 10 N, 1 kK and 0.4 V** for the spring, filament and diode. These are judgements, not datasheet values. The GP proposes only reachable points.
- **An E-stop reset needs an operator name**, and the reset is logged.
- **The red team's scripted adversary** goes through a new `completion=` hook in `LLMClient`, used nowhere else.

### Phase 5
- **`results/` comes from `--full --provider mock`:** 3,680 runs. The baseline rows are final. The LLM rows are mock, and labelled as such in the CSV, the figures and RESULTS.md, together with the check that AutoLab-mock is identical to GP-only on all 120 shared runs.
- **LLM-only got its own sampler** (`--sampler llm`). Its mock and no-LLM rule is: where the top two models disagree most, else the middle of the widest gap.
- **The INR rate is an assumption: 88 per USD** (`--usd-inr` sets the real one). Mock cost estimates use characters / 4 and litellm's bundled price for `anthropic/claude-haiku-5-5`.

### Phase 6
- **The sketch moved to `firmware/arduino_rig/arduino_rig.ino`**, the layout arduino-cli needs. The emulated build's `.hex` and `.elf` are committed, so Wokwi works without compiling.
- **I compiled all three sensor builds** with a temporary arduino-cli in the scratchpad; nothing was installed on your Mac.
- **Firmware watchdog: 300 s.**
- **`--hardware` supports `malus_law` only.**
- **The emulated sensor uses a fixed seed (12345).**

### Phase 7
- **The dashboard drives the digital twin only;** hardware stays on the CLI.
- **Logic lives in `app/lib/`;** the pages only render. A run happens in a background thread, so E-STOP can reach it.
- **The real LLM needs a key, `APP_PASSWORD`, and the password entered.** Per session: at most 10 runs, at most 3 with a real LLM; budgets capped at 30 measurements (20 with a real LLM).

### Environment
- Installed into `.venv`: `hypothesis`, `streamlit` (and its dependencies). Uninstalled: `sympy`.
- `requirements.txt` was updated to match. Phase 8 still has to pin versions.

## Bugs found by the new tests, and fixed

- **The LDR read 0 lux after homing and divided by zero** (found by a property test). The twin homed to 0, outside the LDR's bounds; the inverse-square rail and the filament also homed outside theirs. The twin now parks at the lowest safe position.
- **`math.isfinite(10**400)` raised inside the guard's type check** (found by the red team). The guard now converts values carefully.
- **`tool_take_measurement(None)` crashed while recording the guard's rejection** (found by the red team).
- **The serial driver silently retried a lost reply,** because `TimeoutError` is an `OSError`. It now surfaces as a timeout, which the guard flags.
- **A bare `MOVE abc` would have meant "move to 0°" on the old firmware** (`String.toFloat`). The new firmware rejects it with E_SYNTAX.

## Open questions

- **Random outliers are rarely caught:** 8 of 68 in the Phase 3 sweep, 31 of 230 in the benchmark. Most runs stop at n = 8, when a fit has too few degrees of freedom to call anything an outlier at p = 0.001. Should we add a one-time noise calibration per instrument (repeated readings at one position), so detection can use a known noise level?
- **Hooke with faults converged on cos² in about 1 run in 5** (cos² over 0–20 "degrees" is almost a straight line). A higher `min_points`, or the predictive-stability criterion, would probably fix it at the cost of measurements. Decide after the real benchmark?
- **Are the step limits right** for a real spring loader, lamp and voltage source?
- **Phase 8** (deploy and CI) was outside this run, so it has not started.
