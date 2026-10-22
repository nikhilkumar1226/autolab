# Handoff notes (overnight autonomous run, 2026-10-22)

Written for Nikhil. The detailed "what changed / why / how it was verified" for every phase is in `docs/CHANGELOG.md`; this file is the short version plus everything that needs you.

**Summary:**
- **Done:** Phase 2 (with the AST formula fix), then Phases 3 to 7, each with its checkpoint: tests green, CHANGELOG entry, `Phase N: …` commit, tag, push, ROADMAP boxes.
- **Then Phase 5.1** (your second request): the `llm.check` fix for Groq, and the free-tier subset benchmark `python -m benchmark.run --subset`, which replaces the paid full real-LLM benchmark. **The full real-LLM benchmark is now optional: it needs paid credit.**
- **Tests:** 416 pass offline in about 2 minutes (230 at the start).
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
| 5.1 Free-tier subset, `llm.check` fix | `phase-5.1-free-tier` | `phase-5.1` | 416 | code done and tested offline; the subset run itself is yours (needs your free-tier keys and about 2 h) |

Each branch starts from the previous one, so `phase-5.1-free-tier` contains everything. `main` is still at the initial commit. When you have reviewed it, merging `phase-5.1-free-tier` into `main` is a fast-forward.

## Try it

```bash
source .venv/bin/activate
pytest -q                                             # 416 passed
streamlit run app/Home.py                             # dashboard: Live Lab, Run History, Benchmark, System
python main.py --experiment diode_iv --faults default --nonideal --provider mock
python -m safety.redteam                              # 59 attacks -> results/redteam.csv, 0 unsafe
python -m benchmark.run --quick                       # about 2 minutes -> results/quick/
python -m benchmark.run --subset --dry-run            # the free-tier real-LLM subset: plan and estimate only
```

Results for the report are in `results/`: `RESULTS.md`, `benchmark_*.csv`, `figures/`, `redteam*.{csv,md}`, `phase3_twin_sweep.csv` and the Phase 1 sweeps. **The LLM-method rows in the benchmark are mock.** RESULTS.md says so at the top.

## What you must do yourself

1. **`.env`.** Your `.env` now uses the free-tier chain: Gemini Flash-Lite first, then Groq (`gpt-oss-120b`, `gpt-oss-20b`). That is what the subset uses. I did not touch `.env`. `.env.example` shows both layouts (free tier and Claude first).
2. **Phase 2 checks** (need keys):
   - `python -m llm.check` should print OK for every model in the chain. **Groq should now say OK too**: the ping used to cap the answer at 32 tokens, which a reasoning model (gpt-oss) spends on thinking. Groq then reported `json_validate_failed` with an empty generation. The ping now asks exactly like the agents do. If a Groq line still fails, the line now says "the provider rejected the model's output in JSON mode"; please send me that line.
   - `python main.py --provider litellm`: then check that the run JSON in `runs/` shows the real prompts, answers, tokens and cost.
3. **The real-LLM benchmark: the free-tier subset** (Phase 5.1; this replaces the paid full run for the report):
   - Look first: `python -m benchmark.run --subset --dry-run`. With your `.env` it plans 72 real-LLM runs (llm_only and autolab; malus_law, inverse_square_law, rc_charging, diode_iv; clean, faults, hidden_apparatus; 3 seeds). It estimates about 1,100 calls, ~670k input and ~60k output tokens, and about 1.8 h at 10 requests per minute. It reuses the 480 non-LLM baseline rows already in `runs/benchmark/full/rows`.
   - Run it: `python -m benchmark.run --subset`. It goes one run at a time at no more than 10 requests per minute (`--rpm 5` to go slower) and prints a line per run with the models that answered.
   - **If it stops** with "Stopped: 2 runs in a row found no model available" (exit code 3), a daily limit was reached. Nothing is lost: run the same command again tomorrow and it continues. Ctrl-C is also safe. Do not change `LLM_MODEL`/`LLM_FALLBACKS` halfway: a different chain starts a separate set of rows, so models are never mixed silently.
   - Output: `results/subset/` (RESULTS.md, the CSVs and the figures, all labelled "Reduced subset, real LLM: <models used>"). Then send me `results/subset/`.
   - Answers are cached in `llm_cache/`, so rerunning a finished subset costs nothing.
4. **Optional, needs paid Claude credit: the full real-LLM benchmark.** Use it only if you buy credit and want every experiment and condition with 5 seeds.
   - Put the Claude-first layout from `.env.example` in `.env`.
   - Look first: `python -m benchmark.run --full --provider litellm --dry-run`.
   - Then: `python -m benchmark.run --full --provider litellm --workers 2`. The mock policies made about 6,700 calls, estimated at about $0.6 at litellm's bundled Haiku price; check the Anthropic Usage page after the first runs.
   - The free tiers cannot cover this in reasonable time. The report can rely on the subset instead.
5. **Red team against a real model** (optional, cheap): `python -m safety.redteam --provider litellm --out results/redteam_litellm.csv`. It runs the prompt-injection cases only.
6. **Wokwi** (Phase 6 Done-when):
   1. Install the Wokwi Simulator extension in VS Code.
   2. F1, then **Wokwi: Request a New License**.
   3. F1, then **Wokwi: Start Simulator**, and keep its tab visible.
   4. Run `python main.py --experiment malus_law --hardware --port rfc2217://localhost:4000 --provider mock`.
   5. Screenshot the circuit and save it as `docs/img/wokwi_circuit.png`; the System page shows it.

   Details are in `docs/wokwi.md`.
7. **Dashboard in a browser** (Phase 7): run `streamlit run app/Home.py`, start a run on Live Lab, and press E-STOP once mid-run. I checked this headless only.
   - To use the real LLM there, also set `APP_PASSWORD` in `.env` and type it into the sidebar.
   - Without a password the dashboard always uses mock, even locally. That is deliberate, to protect the credit on the hosted app.
8. **Review my judgement calls** (listed below), in particular the step limits, the default fault schedule and the subset's 10 requests per minute.

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

### Phase 5.1
- **10 requests per minute by default** (`--rpm`). This is a conservative guess for Gemini Flash-Lite and Groq's free tiers, which I could not check against your account. Raise it if your limits allow; the run then finishes sooner.
- **Stop after 2 unavailable runs in a row.** One unavailable run can be a short outage; two in a row is almost always a daily limit. Unavailable runs are never stored, so they are retried.
- **A rate-limited model is skipped for 5 minutes** (after its retries run out) instead of being waited out on every call. Otherwise every call would first sit through Gemini's backoff (two retries, up to 30 s each when the provider asks for it) before reaching Groq.
- **`json_validate_failed` is re-asked like any invalid answer**, in every run, not only in the check.
- **Hidden apparatus is clean physics, and the baselines' clean rows stand in for it** (marked `reused_from = clean` in the CSV). A method without an LLM never reads the description, so rerunning them would give identical rows.
- **Rows are keyed by the model chain**, so if the chain changes (e.g. Gemini removed), the runs start again instead of mixing models. The old rows stay on disk.
- **No learning curves and no equal-budget stage in the subset**, to save calls. The baselines' equal-budget rows from the mock full run are not shown in the subset report, because their budget came from the mock AutoLab.
- **3 seeds for the LLM methods, 20 for the baselines.** RESULTS.md adds a "same seeds only" table (seeds 0–2 for every method) so the comparison can also be made on identical seeds.

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
- **Is 3 seeds per cell enough for the paper?** With 3 seeds, a 95% interval on "correct law" spans most of 0 to 1 unless all three agree. If your daily limits allow, `--llm-seeds 5` on a second day adds seeds 3 and 4 to the same rows (the first 3 are reused).
