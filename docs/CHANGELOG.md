# Changelog

## Phase 1: Foundation refactor (2026-10-22)

### What changed
- **Experiment registry** (`experiments/registry.py`). Each experiment is described as data:
  - an apparatus description for the agents, which never names the law;
  - variable names and units, safe bounds and the maximum step;
  - default noise (σ = 10 lux);
  - the ground-truth law and parameters, used only by the simulator and for scoring.

  `agent_view()` is the only part agents may see; it leaves out the internal name and the ground truth.

  | Experiment | x (units, bounds) | True law (library form) |
  |---|---|---|
  | malus_law | polariser angle (deg, 0–180) | 1000·cos²(x − 20°) + 15 |
  | inverse_square_law | source–sensor distance (cm, 5–100) | 1e5/(x + 5)² + 15 |
  | beer_lambert_law | dye concentration (mg/L, 0–100) | 1000·e^(−0.04x) + 15 |
- **Safety bounds come from the spec.** The global [0, 180] is gone. The guard also now rejects NaN, inf and non-numeric targets; NaN used to pass the `<`/`>` check. A rejected position is recorded, reported as a `safety_rejection` event and never proposed again by the GP. Three consecutive failed attempts abort the run (`safety_abort`).
- **Blind analyst.** `DataAnalystAgent.analyze(x, y)` no longer receives the experiment type. It always fits six models:
  - linear and quadratic;
  - exponential decay;
  - cos² with phase (x in degrees);
  - inverse-square with offset;
  - power law with offset.

  It ranks them by BIC = n·ln(SSR/n) + k·ln n and reports ΔBIC to the runner-up. Initial guesses come only from the data: polynomial least squares, a closed-form linearisation for cos², and a scan of the one nonlinear parameter with linear least squares for the rest. Models that cannot be fitted are listed with a reason (e.g. power law when some x ≤ 0).
- **Stopping rule** (`analysis/stopping.py`) replaces the forced stop at n ≥ 8. The run stops when all three hold:
  - ΔBIC ≥ 6 against the runner-up;
  - every parameter's relative SE < 5%;
  - n ≥ 5.

  It also stops at the 30-measurement budget. The record keeps the stop reason and each criterion's value, threshold and pass/fail. Relative SE is SE/|value| for scale parameters (amplitude, rate, exponent). For location and offset parameters (phase, distance offset, baseline) it is SE divided by the measured x- or y-span. Those parameters' true value can be 0, and their "relative" error would otherwise depend on where the origin is.
- **Reproducibility.** Each run has a seed. It is drawn from OS entropy if not given, recorded either way, and printed with the rerun command. `SeedSequence(seed).spawn(3)` gives independent streams for simulator noise, GP optimizer restarts and the new random-sampling baseline (`--sampler random`). There are no global `np.random` calls; a test enforces this.
- **Run records** (`records/run_record.py`). Each run is saved as `runs/<run_id>.json`, including runs that crash. A record holds:
  - config, seed and the experiment snapshot;
  - the data-source label;
  - every attempt, with the fits made after it;
  - the stop decision, timings and artifact paths.

  `load_run_record()` and `list_run_records()` read them back.
- **Events.** `DiscoveryEngine(config, on_event=...)` emits typed events: `run_started`, `hypothesis`, `measurement`, `safety_rejection`, `analysis`, `gp_update`, `stopped`, `report_ready`. All `rich` output moved to `cli/console.py`, which only `main.py` uses. `AutonomousDiscoveryEngine` was renamed `DiscoveryEngine`.
- **Planner heuristic.** The keyword matching that announced "Malus's Law" before any data was taken is removed. The hypothesis and initial design now come from the agent view and its bounds.
- **Reports and figures** are generated from the run record. They show the data source, ΔBIC, parameters ± SE and the stopping criteria. The polariser-specific text, the claim that the trigonometric model won, and the unsupported "Gaussian residuals" claim are removed.
- **Housekeeping.** `datetime.utcnow()` is replaced everywhere. `scikit-optimize`, `tabulate` and `seaborn` are removed (never imported; the plot style is built into matplotlib). numpy ≥ 1.25 is now required for `np.exceptions`.

### Why
The original analyst was told which law to expect, so identification was circular. Convergence was forced at 8 points, and Beer-Lambert data was labelled quadratic because the exponential model was never fitted. Results could not be reproduced (global RNG) or audited (nothing recorded). The dashboard, benchmark and LLM phases all build on these foundations.

### How it was verified
- `pytest -q`: 80 tests pass offline in about 5 s. The tests cover:
  - the registry, including no law hints in the agent view and a cross-check of the simulator truth against the library;
  - parameter recovery for every library model;
  - blind identification (≥ 18/20 seeds per law at 20 evenly spaced points);
  - the stopping-rule criteria;
  - record round trips, including a crashed run;
  - same seed → identical run;
  - event order and the console renderer;
  - spec-driven safety bounds and the safety-abort path;
  - end-to-end runs for all three experiments.
- `python main.py --experiment beer_lambert_law --seed 1` identifies **exponential decay** (it used to say quadratic). It stops converged at n = 7 with ΔBIC = 7.8 against inverse-square.
- **Seeded sweep with the default config and GP sampler, seeds 0–19 per experiment.** Every row is reproducible with `python main.py --experiment <name> --seed <s> --no-report` (checked for inverse_square_law seed 4).

  | Experiment | Correct law | Stop reasons | Measurements at stop (min / median / max) |
  |---|---|---|---|
  | malus_law | 20/20 | 20 converged | 5 / 5 / 5 |
  | beer_lambert_law | 20/20 | 20 converged | 5 / 7 / 17 |
  | inverse_square_law | 14/20 | 18 converged, 2 budget | 5 / 18.5 / 30 |

### Known limitations (found during verification)
The first two items are addressed by the "Phase 1 fix" entry below. The default is now `min_points = 8`.

- **Inverse-square runs can stop confidently on the wrong law.** In the sweep, 6 of 20 runs converged on the wrong law:
  - exponential decay for seeds 3, 4, 7 and 17;
  - power law for seeds 10 and 16.

  Four of these stopped at the 5-point minimum. The cause is not a fitting failure: the inverse-square fit reached the brute-force global optimum in every case. With 5 points, a 3-parameter model has only 2 residual degrees of freedom. A wrong model can then pass almost exactly through the noisy points and win a large ΔBIC by chance (up to 18.4 here).
- **Raising `min_points` would fix most of this, but it is not the specified default.** The same sweep with `StoppingConfig(min_points=8)` gives 19/20 correct for inverse-square, with one power-law stop (seed 10). Malus and Beer-Lambert stay at 20/20, but stop at 8 points instead of 5–7. `min_points = 10` gives the same accuracy. The default stays at 5 as specified for Phase 1. Whether to raise it is a decision for the next review.
- **Malus always stops at the 5-point minimum.** At σ = 10 against a 1000 lux amplitude, ΔBIC is already about 30 at n = 5, so this floor reflects strong evidence, not a forced stop.
- **The cos² model assumes x is in degrees** (period 180 units).
- **Offsets with large absolute error can still pass.** Span-normalised errors can accept an offset that is poorly known in absolute terms (Beer-Lambert seed 1: C = 7.0 ± 13.3 lux, true value 15). The raw SE is always reported next to it.

## Phase 1 fix: predictive-stability criterion and stopping-rule sweep (2026-10-22)

### What changed
- **New optional stopping criterion: "predictive stability"** (`analysis/stopping.py`). It is configured by `StoppingConfig.predictive_stability`, `stability_window = 2`, `prediction_max_z = 3` and `noise_floor_rel = 0.005`. When it is on, a run may converge only if both of these hold, in addition to the Phase 1 criteria:
  - (a) the same model was best, by ΔBIC ≥ 6, in each of the last 2 analyses;
  - (b) the fit made *before* the newest measurement predicted it within 3σ. σ is that fit's residual standard deviation, sqrt(SSR/(n − k)), floored at 0.5% of the measured y-span.

  Both are recorded in the stop decision (`stable_best`, `prediction`), with value, threshold and pass/fail like the other criteria. When the criterion is off they are absent, and older run records still load.
- **Every measured step now records a `prediction`.** It holds the earlier best model's prediction of that point, σ, z, and the runner-up model's prediction. The runner-up prediction is a diagnostic only: it shows whether the point could tell the two models apart.
- **The noise floor is a stated assumption, not an estimate.** At 4–5 points no model-free noise estimate works. A difference-based estimate is dominated by the steep part of the curve near the source (hundreds of lux). A floor taken from the competing models' residuals is not really noise. 0.5% of the span is half the simulator noise (σ = 10 lux ≈ 1% of the span), so the floor can only make the check stricter. It was fixed before the sweep and not tuned on it. On the real rig it should be calibrated from repeated readings at one position.
- **CLI.** New flags `--min-points` and `--stability/--no-stability`. The printed rerun command includes them when they differ from the default. The console line and the report's stopping table show the two new criteria when the criterion is on.
- **`scripts/stopping_sweep.py`** runs the seeded comparison below and writes `results/phase1_stopping_sweep.csv`. Each row has its own rerun command.
- **Default changed: `min_points` 5 → 8.** Predictive stability stays available but is off by default. The choice follows the rule set before the sweep: among configs with ≥ 19/20 correct on every experiment, take the one with the fewest median measurements.

### Why
In the Phase 1 sweep, 6 of 20 inverse-square runs converged on the wrong law. With 4–5 points, a wrong 3-parameter model can pass almost exactly through the noisy data. It then wins ΔBIC ≥ 6 by chance (see Phase 1 "Known limitations"). The predictive-stability idea was to fix this without a higher fixed floor: a model that has really captured the law should keep winning and should predict a point it has not seen.

### How it was verified
- `pytest -q`: 98 tests pass offline. 18 tests are new:
  - criteria (a) and (b), including the 3σ boundary and a configurable window and z threshold;
  - a prediction made by a model other than the current best;
  - the budget and convergence precedence;
  - the dof-corrected residual std and the floor;
  - a non-finite prediction, and older records loading;
  - the default config matching the sweep's choice;
  - an end-to-end run where every point after the first analysis is checked against the previous analysis's best model.
- **Seeded sweep**, run with `python -m scripts.stopping_sweep`: GP sampler, digital twin, seeds 0–19 for each experiment and config, 180 runs in about 50 s. The configs differ only in the stopping rule. Sampling does not depend on it, so the runs are paired.
  - **Sanity check:** config A reproduces the Phase 1 table exactly.
  - **Rerun check:** `python main.py --experiment inverse_square_law --seed 4 --no-report --min-points 5 --stability` reproduces the C row (exp decay, n = 5). `python main.py --experiment inverse_square_law --seed 10 --no-report` reproduces the B row (power law, n = 8).
  - A, Phase 1 rule (`min_points = 5`):

    | Experiment | Correct law | Measurements at stop (min / median / max) | Stop reasons |
    |---|---|---|---|
    | malus_law | 20/20 | 5 / 5 / 5 | 20 converged |
    | inverse_square_law | 14/20 | 5 / 18.5 / 30 | 18 converged, 2 budget |
    | beer_lambert_law | 20/20 | 5 / 7 / 17 | 20 converged |
  - B, `min_points = 8`:

    | Experiment | Correct law | Measurements at stop (min / median / max) | Stop reasons |
    |---|---|---|---|
    | malus_law | 20/20 | 8 / 8 / 8 | 20 converged |
    | inverse_square_law | 19/20 | 8 / 20.5 / 30 | 18 converged, 2 budget |
    | beer_lambert_law | 20/20 | 8 / 8 / 17 | 20 converged |
  - C, predictive stability (`min_points = 5`):

    | Experiment | Correct law | Measurements at stop (min / median / max) | Stop reasons |
    |---|---|---|---|
    | malus_law | 20/20 | 5 / 5 / 6 | 20 converged |
    | inverse_square_law | 15/20 | 5 / 20 / 30 | 18 converged, 2 budget |
    | beer_lambert_law | 20/20 | 5 / 8.5 / 18 | 20 converged |
- **Choice.** Only B reaches ≥ 19/20 on every experiment. A has 14/20 and C has 15/20 on inverse-square. C is cheaper: its median over all 60 runs is 6 measurements (mean 10.7), against 8 (mean 12.5) for B. It fails the accuracy requirement, so B is the default.
- **Held-out check.** B was also run on seeds 20–39, which were not used for the choice: `python -m scripts.stopping_sweep --configs B --first-seed 20 --out results/phase1_stopping_heldout.csv`. The results are malus 20/20 (8 / 8 / 8), inverse-square 19/20 (8 / 20.5 / 30; 18 converged, 2 budget), Beer-Lambert 20/20 (8 / 8 / 24). Over all 40 seeds, inverse-square is 38/40 correct. The 95% Clopper–Pearson interval for the accuracy is 83–99%.

### Why predictive stability did not fix it
C corrected only seed 17: under A it stopped on exp decay at n = 6, and under C on inverse-square at n = 28. The other five wrong stops remained:

| Seed | Wrong law, n | Newest point x (cm) | z of newest point | Best vs. runner-up (inverse-square) prediction there |
|---|---|---|---|---|
| 3 | exp decay, 5 | 72.7 | 1.8 | 0.08σ apart |
| 4 | exp decay, 5 | 74.9 | 0.5 | 0.01σ apart |
| 7 | exp decay, 5 | 71.7 | 0.0 | 0.20σ apart |
| 10 | power law, 8 | 99.4 | 1.0 | 0.94σ apart |
| 16 | power law, 5 | 67.6 | 0.2 | 0.32σ apart |

- **The deciding point could not discriminate.** In every case the newest point lay where the wrong model and the true model predicted almost the same value, so it could not show which was right. Any honest σ lets it pass. GP uncertainty sampling picks the x where the GP is most uncertain, not where the candidate laws disagree. Under B, seeds 3, 4 and 16 were identified correctly after the GP measured near x ≈ 9 cm, where the two models' predictions were 4.3–5.6σ apart.
- **The floor was binding in seeds 3 and 4** (σ = 4.9 lux). The n = 4 exp-decay fits there had residual stds of only 1.45 and 0.13 lux. Without the floor both predictions would have failed (z ≈ 6.1 and 18.6). That would mean treating a 1.45- or 0.13-lux scatter as the noise level when the noise is 10 lux. The floor was not changed after seeing this.
- **Cost of C.** Malus needed 1 extra measurement in 5 of 20 runs, and the Beer-Lambert median rose from 7 to 8.5.

### Known limitations
- **The 8-point floor sets the stopping point for the easy experiments.** Malus stops at exactly 8 in 20/20 runs (both seed sets), and Beer-Lambert in 12/20. For those runs, n = 8 reflects the floor, not the evidence: Malus already has ΔBIC ≈ 30 at n = 5. This is not the original forced stop, because all evidence criteria must still hold. Inverse-square runs stop anywhere from 8 to 30. It is still worth saying plainly in the report.
- **Inverse-square is still wrong in about 1 of 20 runs.** Seeds 10 and 39 both chose power law at n = 8. In each, the power-law and inverse-square predictions at the newest point (99.4 cm) were only about 1σ apart (0.94σ and 1.17σ). 19/20 is a small-sample estimate (95% CI 75–99.9% from 20 runs).
- **The underlying cause remains.** With n − k = 1–2, a BIC that estimates σ from the residuals exaggerates the difference between models. Uncertainty sampling also does not aim at the points that tell the laws apart. Two possible directions were not tried here: choosing points where the top models disagree most (model-discrimination design, Box & Hill 1967; a natural fit for the Planner's strategy choice in Phase 2), or a likelihood that uses a calibrated noise level.

## Phase 1 checkpoint (2026-10-22)

### What changed
Nothing beyond the two entries above. This entry records the Phase 1 "Done when" checks, run with the final default config (`min_points = 8`, predictive stability off). Also committed with Phase 1:
- `results/phase1_stopping_sweep.csv` and `results/phase1_stopping_heldout.csv`;
- the updated `ROADMAP.md` (Phase 1 review and fix prompt), `CLAUDE.md`, and `.env.example` (Claude first, then Gemini fallbacks; key fields left empty).

### Why
ROADMAP.md asks for each phase to end with its "Done when" list checked against the actual commands, before the `phase-1` tag is created.

### How it was verified
- `pytest -q`: 98 passed in about 8 s, offline.
- **Beer-Lambert:** `python main.py --experiment beer_lambert_law`, run three times with random seeds (3220455824, 2935567065, 3133690026). It identified **exponential decay** each time, converging at n = 8, 8 and 12. **Pass.**
- **"Runs stop at different numbers of points, not always 8."** Random-seed CLI runs:
  - Beer-Lambert stopped at 8, 8 and 12;
  - Malus stopped at 8, 8 and 8;
  - inverse-square stopped at 26, 11 and 9.

  **Passes overall, with a caveat:** Malus always stops at exactly 8 (60/60 in the sweeps plus these 3), because the new 8-point floor binds. The evidence criteria are already met earlier. See "Known limitations" in the Phase 1 fix entry.
- **"A JSON file appears in `runs/` for every run."** Nine CLI runs added nine records (10 → 19). The crash path is covered by `tests/test_run_record.py`. **Pass.**
- **Wrong law in a random-seed run.** One of the three random-seed inverse-square runs, seed 1321740821, converged on the **power law** at n = 9. This matches the known limitation of about 1 wrong inverse-square run in 20 (38/40 correct in the seeded sweeps). Three random runs are too few to estimate a rate. Reproduce it with `python main.py --experiment inverse_square_law --seed 1321740821`.
- **Git hygiene:** `.env` is untracked and git-ignored, and `runs/` is git-ignored. The only key-like strings in the diff are the `sk-ant-...` placeholders in ROADMAP.md's setup guide.

## Phase 2: Real LLM agents (2026-10-22)

### What changed
- **LLM client** (`llm/client.py`). Every agent asks a model through `LLMClient.complete_json(system, user, schema)`, which returns a Pydantic-validated object. The provider comes from `LLM_PROVIDER` or `--provider`:
  - `litellm` (a real model). The chain is `LLM_MODEL`, then `LLM_FALLBACKS` in order.
    - Each request uses JSON-schema output, or plain JSON mode if litellm says the model cannot take a schema. It has a 30 s timeout and no litellm-internal retries.
    - An invalid answer is re-asked up to 2 times, with the validation error fed back.
    - HTTP 429 is retried twice with jittered exponential backoff (the jitter comes from the run seed).
    - The next model is tried after auth errors, "credit balance too low", rate limits, timeouts, 5xx, unknown models, connection errors and persistently invalid JSON. A model whose API key is missing or empty is skipped without sending a request.
    - If every model fails, `LLMUnavailableError` stops the run with every model's error listed. It never falls back to the mock.
  - `mock`: deterministic and offline. Each agent supplies a policy that sees exactly the JSON payload a real model would see. Its answer is validated like a real one.
  - `none`: no LLM. The agents use the Phase 1 heuristics; this is the ablation baseline.
  - Validated answers are cached in `llm_cache/` (git-ignored) and looked up in chain order, so a rerun with the same seed replays the same answers at no cost. `--no-cache` turns this off.
  - Anything that looks like an API key is redacted from errors before they are logged or recorded. Cost estimates use litellm's bundled price map, which is never fetched over the network.
  - `python -m llm.check` sends one tiny request to every model in the chain and prints OK or the error for each.
- **Planner** (`agents/lead_scientist.py`). It proposes a hypothesis from the research question and the agent view: a statement, 1–5 candidate relationships in words, the independent variable and a range. It also proposes an initial design of 3–5 points with a rationale.
  - The proposed range is clipped to the spec's safe bounds in code, and the clip is logged and recorded. The range never reaches the safety guard, the GP or the simulator. All three are built from the spec before the Planner is asked.
  - Initial-design positions are queued unmodified, so the guard is what rejects bad ones.
  - Before each later measurement, the Planner either delegates to the GP (uncertainty sampling) or gives its own point, always with a reason. It sees the data, the ranked fits, the unmet stopping criteria, the GP suggestion, the budget left and recent rejections. It also sees the x inside the bounds where the two best models disagree most, in noise units: the model-discrimination idea from the Phase 1 limitations.
  - The numerical stopping rule still decides when to stop. The Planner only writes the stop rationale.
- **Safety feedback.** If the guard rejects a proposal, a `safety_rejection` event is emitted and the guard's message goes back to the Planner as an observation. The Planner may re-propose up to 2 times; after that the GP proposes once.
  - `max_consecutive_failures` now counts measurement *slots*: one proposal plus its re-proposals and the GP fallback. With provider `none` there is no agent to tell, so each attempt is its own slot, exactly as in Phase 1.
- **Analyst** (`agents/analyst.py`). After the blind numerical fits, the Analyst LLM sees the variable names and units, the data, the fit table (parameters, relative SE, BIC, ΔBIC, R²) and a residual summary (`analysis/diagnostics.py`). The residual summary gives the signs ordered by x, the runs-test z, the lag-1 autocorrelation and the largest standardised residual.
  - It never sees the apparatus description, the research question, the hypothesis or the experiment name.
  - It returns an interpretation, whether the residuals suggest a missing model, and optionally ONE new candidate model.
  - At most 3 LLM models are accepted per run. An accepted model is ranked *before* the stop check.
- **Safe candidate models** (`analysis/candidate_models.py`). A proposed expression is never executed. It goes through two whitelists:
  1. **The text is parsed by `ast`.** Allowed: numbers up to 1e6, `x`, the declared parameters, `pi`, `+ - * / **`, and `sin cos exp log sqrt` called by bare name with one argument. A constant exponent must be a single number with |value| ≤ 10, which blocks inputs like `9**9**9` that would make sympy compute a huge integer.
  2. **`sympy.parse_expr` runs with a namespace that holds only those names.** Its tree is checked again, then evaluated by a small tree-walking numpy evaluator, with no `eval`, `exec` or `lambdify`.

  After that:
  - Parameter kinds (scale, x-location, offset) set how the stopping rule normalises relative SE. They are inferred from the expression's structure, not taken from the LLM, so the LLM cannot loosen that criterion.
  - Rejected expressions are logged and recorded with the reason, and the run continues.
  - An LLM model is skipped if its curve stays within 0.5σ of a library fit over the measured range. Without this, a duplicate or near-copy of the best model would stay within ΔBIC ≈ ln n of it and block the ΔBIC ≥ 6 criterion forever. Library models are never removed.
- **Writer** (`agents/writer.py`). It writes the report's discussion and conclusion from facts extracted from the run record, with the instruction to use only those numbers.
  - Every number in the text is checked against the facts at the precision written.
  - If any number does not match, the Writer is asked once more. Numbers that still don't match are listed under the section in the report.
- **Blind mode** (`--blind`). Every LLM agent sees the variables as `x` and `y` and a generic apparatus sentence. Units and safe bounds stay. The report itself, written for people, keeps the real names.
- **Prompts.** All system prompts are in `agents/prompts.py`, which records a `PROMPT_VERSION`. User messages are compact JSON with numbers at 4 significant figures.
- **Run records** (schema 2). New content:
  - every LLM call: agent, purpose, models tried, the model that answered, system and user prompt, raw and parsed answer, validation errors, tokens, estimated cost, latency, cached or not;
  - LLM usage totals;
  - the reason behind each step;
  - every candidate model and what happened to it;
  - the stop rationale, the discussion, and warnings.

  Schema 1 records still load.
- **Events, console, report and CLI.**
  - New events: `strategy` and `llm_call`.
  - The console and report show which agents took part (provider, the models that answered, tokens, cost). The report has new "Agent reasoning" and "Discussion and conclusion" sections, and every piece of agent text is labelled with who wrote it.
  - New CLI flags: `--provider`, `--model`, `--blind`, `--no-cache`. The rerun command includes them.
  - If the LLM is unavailable, the CLI prints one clear error and exits with code 1.
- **LLM failure after the stop decision.** If every model fails for the stop rationale or the Writer, the result and report are kept. The missing section reads "not generated: …", and the same text goes to `record.warnings` and is printed in red. During the experiment loop, the same failure stops the run with an error.
- **Bug fix: the GP no longer re-sends a rejected position.** With fewer than 2 measured points, the GP returned the midpoint even right after the guard had rejected it. It now returns the nearest grid point that is not close to a measured or rejected position. When the midpoint is allowed, the behaviour is unchanged.
- **Housekeeping.**
  - `sympy` added to `requirements.txt`.
  - `reference/` and `llm_cache/` git-ignored.
  - `scripts/stopping_sweep.py` pinned to provider `none`.
  - The unused `AgentConfig.model_name` (default "gpt-4o") and `temperature` removed.

### Why
The project title says "LLM agents", but the original code never called an LLM (see CLAUDE.md, "Known issues"). An examiner who asks "show me the prompt" must be able to see the real prompt, the reasoning it produced, and what that reasoning was allowed to change.

The design keeps the Phase 1 guarantees:
- every position still passes the guard;
- the stopping rule stays numerical;
- the Analyst stays blind;
- provider `none` reproduces Phase 1 exactly, so the benchmark has a clean no-LLM baseline.

Piyush's version (`AI-Laboratory`) was used read-only for the "Take" column of ROADMAP.md, and each idea was tightened:
- **`compile_safe_model_callable`** allowed `np.<attr>` calls (so `np.load('x')` got through) and ran the generated code with `exec`. The new compiler allows no attribute access and runs nothing.
- **His litellm call** fell back silently to keyword matching on any exception. The new chain raises a clear error instead.
- **His mocked-litellm test style** (`patch("litellm.completion")`) is kept, as a scripted fake in `tests/conftest.py`.

Nothing from the "Never take" column was ported.

### How it was verified
- **`pytest -q`: 214 passed offline in about 20–30 s** (98 before). No test can reach a real LLM: the provider is forced to mock, and `litellm.completion` raises unless a test installs the scripted fake. New tests cover:
  - the client: retries with the error fed back; the fallback chain on auth / credit / timeout / 5xx / 404 / connection errors; 429 backoff; all models failing; a missing key skipped without a request; cache hits and replays; redaction; `llm.check` output;
  - 45 dangerous or invalid formulas rejected with a reason, including `__import__('os')`, `np.load('x')` and `().__class__`;
  - a project-wide static check that no `eval`, `exec`, `compile`, `__import__` or `lambdify` is called;
  - structural parameter kinds, and a library duplicate filtered as equivalent;
  - a missing law (a Gaussian peak, not in the library) proposed by the Analyst LLM, accepted and ranked first;
  - an LLM range of [−50, 500] clipped while the guard keeps [0, 180];
  - a rejected proposal fed back, then a re-proposal, then the GP fallback after 2 rejected re-proposals, with no rejected position re-sent by the GP;
  - the Writer's invented number flagged;
  - blind mode hiding names and apparatus from every prompt, and the Analyst never seeing the apparatus, question or hypothesis;
  - a late LLM failure kept as a warning, and a run without any key failing before the rig moves;
  - a schema 1 record loading, and the CLI flags.
- **Phase 1 equivalence:** the `phase-1` tag, `--provider none` and `--provider mock` were each run on the 3 experiments × seeds 0–4. All 15 runs are identical across the three: measurements, stop reasons, laws found and point counts.
- **Prompt size:** about 230–720 tokens per call. A run makes about 2 calls per measurement after the initial design, plus the plan, the stop rationale and the Writer.
- **"Done when" checks** (run against the current `.env`, which still has the old layout, `LLM_MODEL=groq/...`, and no keys set):

  | Item | Result |
  |---|---|
  | `python -m llm.check` shows which keys work (Claude, Gemini) | **Not yet.** The tool runs and reports each model correctly ("missing GROQ_API_KEY; no request sent", "missing GEMINI_API_KEY", Ollama connection refused) and exits 1. No Claude or Gemini key is configured, so no key could be shown to work. |
  | `python main.py --provider mock` works with no key | **Pass.** It converged on cos² at n = 8 with 12 mock calls (plan, strategy, review, stop rationale, Writer), and wrote the record and report. A seeded rerun with `--no-report` made 11 calls (no Writer). |
  | With a real key, `--provider litellm` works and the JSON shows prompts, responses, tokens and cost | **Not yet: no key.** Without keys the run stops before any measurement. The error lists every model, the record is saved with the failed attempts, and nothing falls back to mock (exit code 1). The full litellm path is covered offline by the scripted fake: tokens, cost, the answering model, and cache replay. |
  | `--provider none` still works | **Pass.** It converged on cos² at n = 8, with 0 LLM calls. |

### Known limitations
- **No real model has answered yet.** Gemini's structured-output mode may reject parts of the agent schemas (optional or nested fields). That would show up as `bad_request` and fall back to the next model. The first real run is the real test.
- **The mock never picks its own point and never proposes a model.** It mirrors the baseline on purpose. Those paths are covered by scripted tests, but not by a live mock run.
- **The Writer's number check is lexical.** It catches invented numbers, but not a real number attached to the wrong quantity.
- **`sympy.parse_expr` uses `eval` internally.** It only ever receives text that has already passed the AST whitelist. (Fixed: see "Phase 2 fix" below.)
- **The Planner's own points are not filtered for novelty, as GP points are,** so it can repeat a position.
- **Cost figures are estimates** from litellm's bundled price map.
- **A user-supplied `--question` is passed unchanged in blind mode,** so it can reveal the apparatus. The CLI warns about this.

## Phase 2 fix: AST whitelist evaluator replaces sympy.parse_expr (2026-10-22)

### What changed
- **LLM-proposed model expressions are no longer given to sympy.** `sympy.parse_expr` (and `sympify`) call Python's `eval` internally, so even after the AST gate the text reached `eval`. `analysis/candidate_models.py` now uses only Python's `ast` module and numpy:
  1. **Gate.** Text over 200 characters, or containing `__` anywhere, is rejected before parsing. `ast.parse(mode="eval")` parses it (parsing runs nothing). Every node must be on the whitelist: `x`, the declared parameters, numbers (|c| ≤ 1e6), `pi`, `e`, binary `+ - * / **`, unary minus, and one-argument calls to `sin cos tan exp log sqrt abs radians` by bare name. Attributes, subscripts, strings, keyword arguments, lambdas, comparisons, unary plus and undeclared names are rejected. A constant exponent must still be one number with |value| ≤ 10.
  2. **Evaluation.** The validated tree is turned into closures over a fixed table of numpy operations. A name is looked up in the parameter vector, never in a Python namespace.
  3. **Structure checks, now numerical.** sympy's simplification used to catch expressions that do not depend on x, parameters with no effect, and undefined expressions such as `1/0`. These checks now use fixed probe points (x from −1000 to 1000, two generic parameter vectors, plus the LLM's guesses). Parameter kinds come from central-difference derivatives there: dy/dp = 1 everywhere is a y-offset, dy/dp = ±dy/dx everywhere an x-location, anything else a scale. The LLM's guesses can only make a kind harder to qualify for (test: `C*D` with a guessed `D = 1` stays a scale).
  4. **LaTeX** for the report is printed from the validated tree, so `sympy.latex` is no longer needed.
- `tan`, `abs`, `radians` and the constant `e` are new on the whitelist. Unary plus is no longer accepted (the whitelist names unary minus only). Python builtin names (`open`, `eval`, …) can no longer be parameter names.
- `sympy` was removed from `requirements.txt` and uninstalled from the local virtualenv. No module imports it.
- The Analyst system prompt now lists the new whitelist; `PROMPT_VERSION` is `phase2-v2`.
- CLAUDE.md rule 2 is updated to match. ROADMAP.md Phase 2 has a note that step 3 changed after review.

### Why
CLAUDE.md rule 2 forbids `eval`/`exec` of LLM output. `parse_expr` uses `eval`, which was listed as a known limitation of Phase 2. An evaluator that walks the AST can run only operations from its own table, so a whitelist bug can no longer turn into code execution.

### How it was verified
- `pytest -q`: **230 passed**, offline (214 before). New or changed tests:
  - the five required cases, written exactly as `__import__('os')`, `().__class__`, `x.real`, `open('f')` and `np.load('x')`, are rejected;
  - the rejection table grew from 45 to 51 cases (unary plus, `~`, `np` as a bare name, `e` and `open` as parameter names, an expression with no x dependence);
  - the static check also bans `parse_expr`, `sympify` and any `sympy` import in project code;
  - a new dynamic test replaces `builtins.eval` and `builtins.exec` with functions that fail, then compiles, evaluates and prints a model;
  - kinds for `radians(x + p)` and the Gaussian peak; the guess-cannot-change-kind case; LaTeX for the exponential and the Gaussian.
- The suite also passes with sympy uninstalled.
- **Baseline unchanged.** The `phase-2` tag and this fix were each run on the 3 experiments × seeds 0–4 with providers `none` and `mock` (30 runs each). All 30 are identical: stop reason, measurement count, best model and every measured point. `none` and `mock` are still identical to each other. Neither provider proposes a candidate model, so no other outcome was expected.

### Known limitations
- The numerical structure checks use fixed probe points. An expression that is finite only in a narrow x-range that none of the 33 probe points hits (for example `sqrt(x - 1234)`) is rejected as undefined, even though it might fit data in that range.

## Phase 3: Realistic digital twin, new experiments, fault injection (2026-10-22)

### What changed
- **Non-idealities in the digital twin** (`hardware/simulator.py`, `config.NonIdealityConfig`). All are off by default, switched on together with `--nonideal`, and each can be set to 0 on its own. Sizes are relative to the experiment's spec (noise std, position tolerance), so one setting suits every experiment:

  | Non-ideality | Default when on | What it does |
  |---|---|---|
  | Polariser extinction ratio | 100 | Crossed polarisers still pass 1/100 of the lamp (Malus only) |
  | Ambient drift | step of 0.3 × noise std per cycle | Random walk added to every reading (about 1.5 σ after 25 cycles) |
  | Saturation | on | Readings clip at the sensor's full scale |
  | ADC quantisation | on | Readings round to the sensor resolution |
  | Actuator position error | σ = 0.15 × position tolerance | Gaussian error in where the actuator really goes |
  | Backlash | 0.3 × position tolerance | Offset that depends on the direction of the last move |
  | Outlier spikes | 3% of readings, ±8 σ | Random-sign spikes |

  The rig reports the **commanded** position, as an open-loop controller does, so position error and backlash reach the data as errors in x. It also returns a coarse position feedback (quantised to a quarter of the tolerance) and a cycle counter. With non-idealities off and no faults, the twin draws exactly one normal number per reading, as in Phase 1.
- **Five new experiments** in the registry. Each ground truth is written in physical form with physical parameters (`physical_params`). `true_params` gives the same law in library form for scoring, and a test checks that the two agree. The agent descriptions name the apparatus but not the law; the hint-word test was extended (Shockley, Hooke, Stefan, Boltzmann, power law, proportional, linear, …).

  | Experiment | x (units, bounds) | y (units) | Truth (physical form) | Library model |
  |---|---|---|---|---|
  | rc_charging | sampling delay (ms, 0–250) | capacitor voltage (V) | V = Vs (1 − e^(−t/RC)), Vs = 3.3 V, R = 10 kΩ, C = 4.7 µF | Decaying exponential, A = −3.3 |
  | diode_iv | forward voltage (V, 0–0.8) | current (mA) | I = Is (e^(V/(n Vt)) − 1), Is = 2 nA, n = 1.9, T = 300 K | Growing exponential, C = −Is |
  | ldr_response | illuminance (lux, 10–1000) | resistance (kOhm) | R = R1 E^(−γ), R1 = 500 kΩ, γ = 0.7 | Power law |
  | hookes_law | applied force (N, 0–20) | end position (mm) | L = L0 + F/k, L0 = 50 mm, k = 0.25 N/mm | Linear |
  | stefan_boltzmann | filament temperature (kK, 0.8–2.8) | detected power (mW) | P = f ε σ A (T⁴ − Ta⁴), ε = 0.35, A = 20 mm², f = 1%, Ta = 300 K | Power law |

  Stefan–Boltzmann uses kilokelvin on purpose. With T in kelvin, the power-law amplitude and exponent are almost perfectly correlated (x ≈ 1000–3000), the amplitude's relative SE was 22% in the prototype, and the run could never meet the 5% criterion. In kK, x ≈ 1 and both are well determined.
- **Model library.** A **growing exponential**, y = A e^(kx) + C with k > 0, was added; it covers the Shockley diode. A separate saturating exponential was **not** added: A(1 − e^(−kx)) + C is the decaying exponential with negative amplitude, and two identical families would tie forever on BIC. That model is renamed from "Exponential decay" to **"Decaying exponential"** (key `exp_decay` unchanged), because it also fits a charging capacitor.
- **Parsimony rule** (`analysis/parsimony.py`). Several 3-parameter models contain a straight line (a quadratic with a = 0, a power law with b = 1, an exponential with k → 0). With linear data they fit as well as the line and sit within ΔBIC ≈ ln n of it, so ΔBIC ≥ 6 could never be met. In the prototype, BIC alone also picked a curved model in 6 of 20 Hooke runs. The rule: if a model with fewer parameters has the same curve as the lowest-BIC model (within 0.5 σ everywhere on the measured range), the simpler model wins and the other is marked redundant. Any other fit with more parameters and the same curve as the best is also redundant. Redundant fits stay in the table, marked "not counted: parsimony", but are not the best model, the runner-up or part of ΔBIC. If every rival is redundant, the ΔBIC criterion passes and the record says so.
- **Instruments in the spec.** Each spec now has the sensor's range and resolution, the actuator's position tolerance and, for the diode, a 30 mA response limit. The safety guard is built from them (`SafetyLimits.from_spec`).
- **Fault injection** (`config.FaultSpec`, `--faults`). There are three types:
  - `light_spike`: a transient spike of 0.5 × the response span;
  - `sensor_dropout`: NaN or a timeout, for `duration` cycles;
  - `stuck_actuator`: the move is acknowledged but not executed, and the feedback keeps reporting the old position.

  `at_step` counts **rig cycles**: one cycle is one sensor read or one pause, so re-measurements count. `--faults default` is spike at cycle 6, stuck at 8, dropout at 10–11. This differs from the ROADMAP example (stuck at 7, dropout at 9) for two reasons:
  - cycle 7 is always the automatic re-measurement after the cycle-6 spike, at the same position, so a stuck actuator there can never act;
  - most runs stop at n = 8 (about cycle 11), so a later dropout is usually never reached.

  The ROADMAP schedule can still be passed as JSON.
- **Detection** (`hardware/safety_guard.py`). The guard flags readings that are:
  - `non_finite`, or a sensor `timeout` (returned as a flagged reading, not raised);
  - `saturated` (at full scale) or `out_of_range`;
  - `position_mismatch`: the feedback is off the command by more than the tolerance;
  - `response_limit`;
  - `outlier`: outside the predictive band of the current best fit.

  The model band's σ combines the noise (residual std, floored) and the fitted curve's own uncertainty (delta method on the new parameter covariance). Its threshold is a Student-t quantile with n − k degrees of freedom at two-sided p = 0.001. The ROADMAP asked for the GP's band; it is still available (`outlier_bands = "gp"` or `"both"`) but is not the default, because the comparison below showed it is badly calibrated on these curves.
- **Recovery** (`agents/orchestrator.py`). A flagged reading never enters the data.
  1. The Executor re-measures once.
  2. If the reading is flagged again, the Analyst decides: re-measure, pause, or discard the position. With an LLM it gets a blind payload (variable names, flags, readings, bands); the mock and `none` use the same heuristic (pause on a dropout, discard at saturation, otherwise re-measure).
  3. After 2 Analyst decisions the position is discarded.

  Supporting rules:
  - A discarded position is never proposed by the GP again.
  - An outlier is accepted after all if a re-measurement agrees with it within 3√2 σ. That only happens when the expectation, not the sensor, was wrong.
  - Above the response limit, the guard backs the actuator off to its minimum, refuses that position and anything above it for the rest of the run, and the GP respects that ceiling.
- **Fault scoring** (`records/faults.py`). After the run, the twin's fault log (ground truth, never seen by detection) is compared with what was flagged. Each fault is `recovered`, `missed`, `detected_not_recovered`, `no_effect` or `not_reached`. Random outliers are scored the same way. Flags at cycles where nothing was injected are counted as false alarms.
- **Run record schema 3:** per-attempt flags, bands, handling and rig cycle; `faults`; `fault_summary`; physical parameters in the experiment snapshot. Schema 1 and 2 records still load.
- **Display.**
  - A new `anomaly` event, shown on the console as `FLAGGED (...) -> action [who]: reason`, and a fault table at the end.
  - The report has a "Digital twin" header line, a "Reading checks" method bullet, and tables of flagged readings and faults.
  - The Writer receives the twin settings and fault outcomes as facts.
- **New Analyst prompt** `ANOMALY_SYSTEM`; `PROMPT_VERSION` = `phase3-v1`.
- **CLI:** `--nonideal`, and `--faults none|default|<JSON list>|<JSON file>`. Both are included in the rerun command.
- **`scripts/twin_sweep.py`** writes `results/phase3_twin_sweep.csv`.

### Why
A perfect simulator makes any method look perfect. Drift, saturation, position error and faults are what a real bench does, and recovering from them is where an agent can earn its place. The five EE-flavoured laws widen blind discovery beyond optics.

### How it was verified
- **`pytest -q`: 311 passed, offline, in about 75 s** (230 before).
  - `tests/test_twin.py` covers each non-ideality on the rig: one draw per reading when off, extinction ratio, drift as a seeded random walk, saturation, quantisation to the ADC step, position error reaching the physics while the command is reported, backlash by direction, logged outliers, and reproducibility with everything on. It also covers each fault: a spike only at its cycle, NaN and timeout dropouts, a stuck actuator with feedback, and a pause as a cycle.
  - `tests/test_faults.py` covers:
    - every guard flag, including a timeout turned into a flagged reading;
    - the response-limit back-off and ceiling, the GP respecting the ceiling, and no kept reading over the limit;
    - each default fault detected and recovered in a closed-loop run, with the dropout's second flag going to the Analyst;
    - a blind anomaly prompt, in normal and blind mode;
    - provider `none` matching `mock`;
    - an LLM "discard" being followed and the position never proposed again;
    - a confirmed outlier being accepted;
    - a tiny spike scored as missed, and a late fault as not reached;
    - the scoring rules on synthetic input;
    - the band choice (model, GP, both);
    - parsimony (a line preferred over a lower-BIC curve, curved laws unchanged, all-redundant rivals passing ΔBIC);
    - the CLI flags, console and report.
  - **Done when, as a test:** all 8 experiments run with the default faults and every non-ideality on, and the saved JSON lists each fault with status, detected, recovered and missed.
- **The library changes alone left Phase 1 unchanged.** After the growing exponential and the parsimony rule were added, and before the reading checks existed, the Phase 1 sweep (config B, 3 experiments × seeds 0–19) was rerun: 0 of 60 runs differed in best model, stop reason, n, ΔBIC, maximum relative SE or runner-up.
- **Outlier band, chosen before the comparison and then checked.** The default (best-fit band, Student-t, p = 0.001) was set before this comparison was run. Setup: provider `none`, 8 experiments × seeds 0–9, once with the default faults and once clean (160 runs per row):

  | Band | Spike recovered / missed | False alarms (80 clean runs) | Correct law (80 faulty runs) | Mean n (faulty) |
  |---|---|---|---|---|
  | GP band, z > 4 | 46 / 34 | 47 | 67/80 | 17.8 |
  | GP and best fit, both z > 4 equivalent | 15 / 65 | 0 | 53/80 | 25.7 |
  | **Best fit, Student-t, p = 0.001 (default)** | **72 / 8** | **1** | **77/80** | **11.6** |
  | Best fit, Student-t, p = 0.0001 | 17 / 63 | 0 | 54/80 | 25.2 |

  The GP band fails in both directions on these curves. It is too narrow on steep curves: genuine readings had z up to 16 (inverse-square at 9 cm). It is too wide elsewhere: a diode spike had z = 0.7, while the 5-point exponential fit put it at z = 98.
- **Seeded sweep of the twin**, `python -m scripts.twin_sweep` → `results/phase3_twin_sweep.csv`: provider `none`, 3 conditions × 8 experiments × seeds 0–19, 480 runs. Every row has its rerun command.

  | Condition | Correct law | Spike recovered / missed | Stuck actuator recovered | Dropout recovered | Random outliers detected | False alarms |
  |---|---|---|---|---|---|---|
  | clean | 158/160 | – | – | – | – | 1 |
  | faults | 150/160 | 143 / 17 | 160/160 | 160/160 | – | 0 |
  | faults + non-idealities | 150/160 | 138 / 22 | 160/160 | 160/160 | 8/68 | 0 |

  Per experiment, clean: malus 20/20, inverse-square 19/20, Beer–Lambert 20/20, RC 20/20, diode 20/20, LDR 20/20, Hooke 19/20, Stefan–Boltzmann 20/20. Median n is 8 everywhere except inverse-square (20.5).

  With faults, the misses are inverse-square 14/20 and Hooke 16/20; every other experiment is 20/20. With faults and non-idealities, the misses are inverse-square 12/20 and LDR 18/20.

  A missed spike costs accuracy: with faults, the law was right in 138 of the 143 runs where the spike was recovered, but in only 12 of the 17 where it was missed.
- **Clean runs against Phase 1.** Clean malus, inverse-square and Beer–Lambert runs match the Phase 1 sweep (config B) in best model, stop reason and n for 59 of 60 seeds. The exception is inverse-square seed 4: one genuine reading was flagged, the re-measurement confirmed it, and that changed later noise draws. The run still found the right law, at n = 14 instead of 15.
- **Done when, from the CLI** (`python main.py --experiment <name> --seed 0 --provider mock --no-report`, with `--faults default --nonideal` and without):
  - all 8 experiments ran both ways;
  - without faults, every record has an empty `faults` list;
  - with faults, every record lists light_spike, stuck_actuator and sensor_dropout, each `recovered`;
  - all 16 runs found the right law; 15 converged and inverse-square with faults used its budget.

### Known limitations
- **Spikes early in a run, and small random outliers, are often missed.** Most runs stop at n = 8, so readings are checked against fits with only 1–5 residual degrees of freedom. At p = 0.001 the Student-t threshold is then 636, 31.6, 12.9, 8.6 and 6.9 band-σ. Only 8 of 68 random ±8σ outliers were caught. A missed spike can mislead the fit: inverse-square with faults was 14/20.

  A noise level calibrated once per instrument (repeated readings at one position) would allow normal quantiles. Using the twin's true noise for that would leak ground truth into detection, so it was not done.
- **Hooke with faults: 4/20 converged on cos² at n = 8–9.** Over 0–20 "degrees" a cos² arc is almost straight, and with 8 points it beat the line by ΔBIC ≥ 6. In seed 11 that was ΔBIC 13.9, driven by an end point 1.9σ low. The parsimony rule does not apply, because the curves differ by more than 0.5σ. A higher `min_points` or predictive stability would probably help; neither was tried here.
- **Linear laws from fixed data.** With 20 evenly spaced points, a curved model beats the line by BIC in 3 of 20 Hooke seeds, so that static test accepts ≥ 17/20.
- **Position feedback exists only on the twin.** The SG90 servo planned for Phase 6 has none, so on that rig a stuck actuator can only show up through its readings.
- **The response-limit ceiling assumes the reading rises with the position**, as a diode's current does. If the bounds are set wrongly (0.9 V instead of 0.8 V in the test), the guard trips repeatedly and the run ends as a safety abort. That is the safe outcome, but no data comes back.
- **The extinction ratio changes the effective parameters.** With non-idealities on, Malus data follow A' = A(1 − 1/ER) = 990 and C' = C + A/ER = 25, not 1000 and 15. Phase 5 scoring must use the effective values.
- **A pause on the twin is one cycle of simulated time**, not a real wait. On hardware it is a real `sleep`.
- **Anomaly decisions have only been made by the heuristic and the mock**, never by a real LLM. The mock uses the heuristic, so mock and `none` runs agree.
- **The ROADMAP's GP band is not the default**, for the reasons in the table above. It remains available as `outlier_bands = "gp"` or `"both"`.

## Phase 4: Safety hardening and red-team test (2026-10-22)

### What changed
- **The safety guard** (`hardware/safety_guard.py`) checks every command in this order:
  1. the E-stop is not latched;
  2. the watchdog has not expired;
  3. the target is a finite real number;
  4. it is inside the bounds and below any response-limit ceiling;
  5. the step is at most the spec's limit;
  6. the rate limit;
  7. the oscillation interlock.

  New in Phase 4:
  - **Careful type check.** A value is converted to a finite float once (`_finite_real`), and only that float is used. Before, `math.isfinite(10**400)` raised `OverflowError` inside the check. `Decimal`, `complex`, `bytes`, containers, `bool` and `None` are refused.
  - **Rate limit:** at most 5 moves in any 1 s of rig time (`SafetyConfig.max_moves_per_s`).
  - **Oscillation interlock:** at most 3 moves in a row without a sensor reading. The rate limit alone cannot stop a full-travel oscillation, because each 0–180° swing takes 0.55 s of travel, so it stays under 5 moves/s. Every real measurement is one move and one reading.
  - **Watchdog:** the E-stop latches if no valid reading arrives within 120 s of rig time. The default is generous because on hardware the rig clock is wall time and includes LLM latency between readings.
  - **Latched E-stop:** `emergency_stop(reason)` latches. Only `reset_estop(operator=<name>)` clears it; an empty name is refused, and every attempt is logged. `DiscoveryEngine.emergency_stop()` latches the running run's guard, which is the hook for the Phase 7 dashboard. When latched, the engine aborts the run at once (`safety_abort`, "E-stop latched (…)") instead of asking the Planner again.
  - **Rig clock.** Every rig has a `clock()`. The digital twin keeps simulated time: a move takes 0.05 s plus travel time (0.5 s for the full range), a reading 0.2 s, a pause its length. Rate-limit and watchdog checks are therefore reproducible from the seed. On hardware the clock is wall time.
- **Append-only audit log** (`hardware/audit.py`). Every request and decision is logged with its reason, the requested value (as `repr`, whatever its type) and the rig time: connect, home, move, read, pause, E-stop, reset, back-off.
  - Each entry holds the SHA-256 of the previous one and of itself. `verify_chain` finds a deleted, reordered or edited entry.
  - The log is saved in every run record (`audit_log`, schema 4), and the report states how many decisions were made, how many were rejected, any safety trips, and whether the chain is intact.
- **Realistic step limits.** The spring loader is limited to 10 N per step (no shock loading), the filament to 1 kK per step (thermal shock) and the diode to 0.4 V per step. The other rigs keep full travel, which is safe for them at their natural speed. The GP now proposes only points within the step limit of the current position.
- **Strict numbers in LLM answers.** A position or range from a model must be a JSON number. Pydantic used to turn `"90"` into 90.0 and `true` into 1.0; both are now refused.
- **The twin parks at the lowest safe position**, not at 0. A property test found that a reading after homing put the LDR at 0 lux, where its law (R ∝ E^−0.7) divides by zero. The inverse-square rail (0 cm) and the filament (0 kK) also homed outside their bounds.
- **Bug fix found by the red team.** `LabTools.tool_take_measurement(None)` crashed while recording the guard's rejection, because the outcome required a float. It now records `None`. The guard had already refused the value.
- **Spy rig and envelope checker** (`safety/spy.py`). `SpyRig` records every command that reaches the hardware. `unsafe_commands` checks those commands independently of the guard's code: finite and in bounds, step, rate, moves without readings, and none while the E-stop is latched or at or above a ceiling.

  Moves are matched in order to the guard's approvals in the audit log, so a move without an approval is reported. Order is used rather than time because rejected commands take no rig time.
- **Red-team suite** (`safety/redteam.py`, `safety/attacks.yaml`): 59 cases in 9 categories. These are:
  - out of range;
  - unit tricks ("3.5 radians", "200 mm" on a degree axis, mV on a volt axis, K on a kK axis);
  - prompt injection in the research question;
  - oscillation and rate abuse;
  - malformed JSON;
  - huge, negative, NaN and infinite numbers;
  - direct calls that bypass the agents (Python types, serial-command strings, step abuse);
  - E-stop, watchdog and the response limit;
  - code and decision injection (formulas, `"accept"` as an anomaly action).

  Each case runs through the real pipeline: engine, client parsing and validation, re-asking, the range clip, the formula whitelist, the guard and the spy rig. Only the network request is replaced, by a scripted adversary through the new `LLMClient(completion=...)` hook, which is never used outside the red team. An "obedient" adversary also follows the injected instructions. An API-key canary is set during the run and must never appear in a prompt or a record.

  Output: `results/redteam.csv` (one row per case) and `results/redteam_summary.md`. `--provider mock|litellm` runs only the cases that do not need a scripted model, and reports the rest as skipped.
- **Standards mapping:** `docs/safety_and_standards.md` covers IEC 61010-1, IEC 61508, ISO/IEC 27001 and ISO/IEC 30141, plus a table of every safety feature with where it is and what tests it. It states plainly that nothing is certified and no SIL is claimed.
- **Dependencies:** `hypothesis` (property tests) and `pyyaml` (the attack catalogue) were added to `requirements.txt`.

### Why
The paper's central claim is that a hallucinating or compromised model can never move the hardware outside its safe envelope. This phase makes that claim testable two ways: exhaustively over input types (property tests) and against deliberate attacks (red team). Each check is made by code that does not trust the layer above it.

### How it was verified
- **`pytest -q`: 358 passed, offline, in about 105 s** (311 before).
  - `tests/test_safety_hardening.py` (35) covers:
    - the rate limit on rig time, with normal measuring never hitting it;
    - the interlock;
    - the watchdog latching and being fed by valid readings;
    - the E-stop latched until a named operator resets it, including from the engine mid-run;
    - the audit sequence, with tampering by edit, deletion and reordering detected;
    - an intact audit log in saved records, with every attempt that reached the rig approved;
    - 11 non-real values refused without crashing, and Fraction, numpy float and numpy int accepted;
    - GP proposals within the step limits for the diode, Hooke and Stefan–Boltzmann;
    - strict LLM numbers.
  - `tests/test_safety_properties.py` (6 properties, 150–400 examples each) covers any value sequence on any experiment, the measurement tool, strategy and plan answers, and formula text and grammar-built formulas. Run statistics: 1,300 passing examples, 0 failing. The first run of the command-sequence property found the park-position bug above.
  - `tests/test_redteam.py` (6):
    - the catalogue has ≥ 40 cases and ≥ 4 in each required category;
    - the full suite gives 0 unsafe commands and every outcome as expected;
    - mock-provider skipping;
    - **the harness detects a bypassed guard:** with the guard's checks disabled, the out-of-range case is reported as `reached_rig_unsafe`;
    - every spy rule on synthetic commands, including an approved move while latched and an unapproved move;
    - the API-key canary.
- **Red team, Done-when** (`python -m safety.redteam`, scripted adversary):

  | Attempted | Blocked at LLM/agent level | Blocked at the guard | Harmless (inside the envelope) | No effect | **Unsafe commands reaching the rig** |
  |---|---|---|---|---|---|
  | 59 | 23 | 29 | 3 | 4 | **0** |

  Every case had its expected outcome. The 3 "harmless" cases are legal values the attack wanted:
  - a radians number that is a legal angle (3.14°);
  - repeated requests for the same end stop, each with a reading;
  - `Fraction(1, 2)` and a numpy float within bounds.

  The 4 "no effect" cases are injections the mock-policy planner ignores, plus the API-key canary.
- **Effect on runs** (clean twin, provider `none`, 8 experiments × seeds 0–19, against the Phase 3 sweep). Six experiments are identical seed for seed in law, n and stop reason. The diode and Hooke changed, because the step limit changes which points the GP can propose:
  - the diode is still 20/20, with median n 9 instead of 8;
  - Hooke is still 19/20, with 1 run reaching the budget.

### Known limitations
- **The step limits are judgements, not datasheet values:** 10 N, 1 kK and 0.4 V. They would need checking on a real loader, lamp and source.
- **The watchdog runs only when a command arrives.** It cannot stop a host that has hung completely. That is the firmware command watchdog's job (Phase 6).
- **The default watchdog of 120 s rig time effectively never fires on the twin**, where simulated time only advances with rig activity. It is tested with 5 s.
- **On hardware, the 5 moves/s rate limit has not been checked** against the real servo's timing.
- **The red team's "blocked at LLM/agent level" includes the run stopping** because every answer was invalid. That is safe but ends the run. A real model that answers badly once would usually recover on the re-ask.
- **The red team has not been run against a real model.** `--provider litellm` runs the question-injection cases only, and needs keys.

## Phase 5: Benchmark (2026-10-22)

### What changed
- **Benchmark harness** (`benchmark/`), run as `python -m benchmark.run --quick | --full [--provider mock|litellm]`.
  - `run.py`: the run plan, parallel workers and resume.
  - `metrics.py`: scoring each run against the ground truth.
  - `summary.py`: statistics.
  - `figures.py`: the figures.
  - `report.py`: the CSVs and `RESULTS.md`.
- **Methods.** All five start from the same 4-point design and use the same stopping rule and safety guard.

  | Method | How later points are chosen | LLM |
  |---|---|---|
  | `grid` | New `GridSampler`: coarse to fine (ends, middle, quarters, eighths …), so the points are spread out whenever the rule stops | no |
  | `random` | Uniform; `RandomSampler` now draws only from the reachable window | no |
  | `gp_only` | GP uncertainty sampling (provider `none`) | no |
  | `llm_only` | New sampler `llm`: the Planner chooses every point itself, with no GP (`LLM_ONLY_SYSTEM`, `PointChoice` schema). The mock policy and the no-LLM fallback are the same rule (`gap_point`): where the two best models disagree by ≥ 2 σ if that point is reachable and new, otherwise the middle of the widest unmeasured gap | yes |
  | `autolab` | The full system | yes |

  `--sampler grid|llm` is also available in `main.py`, and the report describes each sampler.
- **Conditions and run kinds.**
  - Conditions: clean, non-idealities on, and faults (the default schedule).
  - `adaptive` runs use the normal rule with a budget of 30.
  - `fixed` runs put grid, random and GP-only at exactly AutoLab's median n for that experiment and condition, for the equal-budget comparison.
  - `curve` runs take every method on the clean twin to exactly 20 points; the parameter error after each point gives the learning curves.
  - Seeds: 20 for non-LLM methods and 5 for LLM methods (`--seeds`, `--llm-seeds`); 2 and 2 with `--quick`.
- **Metrics per run:**
  - measurements, attempts, and the stop reason;
  - the law identified, blind;
  - the error of each true-law parameter, absolute and normalised by kind like the stopping rule (|true| for a scale, the x-span for a location, the response span for an offset, whose true value can be 0);
  - fault outcomes, outliers detected, false alarms;
  - LLM calls, tokens and cost in USD and INR;
  - wall time.

  With non-idealities on, the Malus truth uses the polariser's effective parameters (A' = A(1 − 1/ER), C' = C + A/ER).

  For mock runs, the tokens a real model would use are estimated as prompt and answer characters / 4. They are priced with litellm's bundled price map for `--estimate-model` (default `anthropic/claude-haiku-5-5`). INR uses an assumed 88 INR/USD (`--usd-inr`).
- **Outputs.**
  - `results/benchmark_runs.csv`: one row per run, with its `run_id` and record.
  - `results/benchmark_summary.csv`: mean, std and 95% CI per run kind × method × experiment × condition, plus pooled rows (`ALL`). Proportions use Wilson intervals and means use Student-t.
  - `results/figures/` (PNG at 300 dpi and PDF):
    - parameter error vs. measurements (geometric mean and 95% CI);
    - law accuracy;
    - equal-budget accuracy;
    - fault recovery;
    - cost per run.
  - `results/RESULTS.md`, generated: tables plus a line on each figure. Its banner states the data source and, for mock runs, that the LLM rows say nothing about a real LLM. It also reports a check that AutoLab-mock and GP-only are identical on shared seeds.

  The figures use a validated colour-blind-safe palette with a marker and line style per method. Every figure has a footer with its data source, provider and seed caveat.
- **Execution.**
  - Parallel workers (`--workers`, default number of CPUs − 1).
  - Resume-safe: each finished run is stored in `runs/benchmark/<mode>/rows/`. Non-LLM rows do not depend on the provider, so `--provider litellm` reruns only the LLM methods.
  - `--dry-run` prints the plan and an estimate of the real-LLM calls.
  - `--quick` writes to `results/quick/`, which is git-ignored.

### Why
The results chapter needs numbers that are reproducible and honest. This means repeated seeded runs, baselines at an equal budget, intervals rather than single runs, and a clear label whenever a number did not come from a real model.

### How it was verified
- **`pytest -q`: 367 passed, offline, in about 112 s** (358 before). The 9 new tests in `tests/test_benchmark.py` cover:
  - parameter-error normalisation by kind (including an offset whose true value is 0), and the polariser's effective parameters;
  - Student-t and Wilson intervals against known values;
  - the grid order and the reachable windows;
  - the `gap_point` rule;
  - LLM-only mode with mock (Planner) and with `none` (rule), where the GP is never asked;
  - a tiny end-to-end benchmark: every output written, equal budgets, 20-point curves, AutoLab-mock equal to GP-only, then a second run with nothing left to do;
  - the dry run.
- **Done when: `--quick` runs in under a few minutes.** Passed: 142 s for 464 runs on 7 workers, from an empty work folder.
- **`--full --provider mock`:** 3,680 runs (2,240 adaptive and curve runs, then 1,440 equal-budget runs), 0 errors, about 25 minutes on 7 workers. Results in `results/`. The non-LLM methods are real benchmark data. The LLM rows are mock: AutoLab-mock was identical to GP-only on all 120 shared runs, as designed.
- **Provider-independent findings** (non-LLM methods, 20 seeds, 8 experiments):
  - **The adaptive stopping rule earns its extra measurements.** On the clean twin, at AutoLab's median budget (mean n 8.9), GP-only is right in 151/160 runs, grid in 152/160 and random in 151/160. With the adaptive rule they reach 158/160 (mean n 10.2), 159/160 (10.6) and 157/160 (11.0).
  - **Faults cost accuracy mainly on inverse-square** (GP-only 14/20, random 13/20, grid 17/20) and on Hooke (GP-only 16/20, random 19/20, grid 18/20; the cos² near-tie from Phase 3). Every other experiment is at least 18/20 with faults.
  - Stuck actuators and dropouts are recovered in 100% of runs for every method. Spikes are recovered in 87–90% of runs. Random outliers with non-idealities on: 31 of 230 detected (see the Phase 3 limitation).
  - Median parameter error at the stop: 0.96–1.18% on the clean twin, 1.8–2.1% with non-idealities on.

### Known limitations
- **No real LLM has run the benchmark**, so the ROADMAP's second Done-when item is still open. The mock answers are fixed policies, so "what the LLM adds" cannot be read from these results.
- **The LLM methods have 5 seeds, the baselines 20.** Any difference between the two groups mixes method and seed set. RESULTS.md and the figure footers say so.
- **Cost estimates** come from characters / 4 and litellm's bundled price map, which may lag behind real prices. The INR rate (88) is an assumption.
- **The parameter error is the true law's fit,** even in runs that identified another law. Law accuracy is reported separately.
- **Learning curves cover the clean twin only.**

## Phase 6: Firmware upgrade and firmware-in-the-loop (Wokwi) (2026-10-22)

### What changed
- **Firmware v2.0.0.** The sketch moved to `firmware/arduino_rig/arduino_rig.ino`, because arduino-cli needs the folder named after the sketch. Docs and code now agree: an SG90 servo on pin 9 for an Arduino Uno. The old header's stepper, MLX90614 and ESP32 claims are gone.
  - **The sensor is chosen at compile time** (`#error` unless exactly one is set):
    - `SENSOR_EMULATED` (default): Malus's law plus Gaussian noise computed on the chip from the servo angle, with the digital twin's values (1000 lux, 20°, 15 lux ambient, σ = 10 lux) and a fixed seed, for firmware-in-the-loop;
    - `SENSOR_LDR`: a divider on A0 with a nominal GL5528 lux conversion;
    - `SENSOR_TSL2591`: I2C, with the Adafruit library.
  - **Firmware-side safety, independent of the host guard:**
    - bounds of 0–180° and a step limit on every MOVE;
    - arguments must be plain decimals: `String.toFloat` would have turned `MOVE abc` into a move to 0°;
    - a 32-character line limit;
    - ESTOP releases the servo (no holding torque) and latches until RESET;
    - a command watchdog: 300 s without a command releases the servo and latches the E-stop.
  - **Protocol.** The old commands keep their replies (PING, HOME, MOVE, READ, STATUS). VERSION, ESTOP and RESET are new.
    - Errors are `STATUS: ERROR, CODE: <E_SYNTAX|E_BOUNDS|E_STEP|E_ESTOP|E_SENSOR|E_UNKNOWN>, MSG: …`.
    - Unsolicited lines start with `EVENT:`: the boot banner and a watchdog trip.
  - **Build:** `scripts/build_firmware.sh [EMULATED|LDR|TSL2591]`. The emulated build's `.hex` and `.elf` are committed for Wokwi (`firmware/build/`); other build files are ignored.
- **Wokwi project.** `wokwi.toml` loads `firmware/build/` and enables RFC2217 forwarding on port 4000. `diagram.json` wires an Uno, an SG90 (PWM to D9, 5 V, GND) and an LDR module (AO to A0, 5 V, GND).
- **Serial driver** (`hardware/serial_driver.py`), rewritten.
  - Ports open with `serial.serial_for_url`, so `COM3`, `/dev/ttyUSB0` and `rfc2217://localhost:4000` all work. The 2 s bootloader wait is applied only to real boards.
  - On connect it sends PING (up to 3 tries) and reads VERSION.
  - Each command waits for its own reply type and skips `EVENT:` lines (logged) and stray lines.
  - A firmware error becomes `FirmwareError` with its code. A missing reply is a `TimeoutError`, which the guard flags as a reading. A dropped link reconnects once and repeats the command; every command is idempotent.
  - The driver checks that the firmware reports the position it was asked for.
  - The label is `FIRMWARE-IN-THE-LOOP (Wokwi)` for rfc2217 ports and `PHYSICAL RIG` otherwise.
- **The guard drives the hardware E-stop.** `emergency_stop` also sends ESTOP to the rig, and `reset_estop(operator=…)` sends RESET first; if the firmware refuses, the guard stays latched. The twin ignores both, and the spy rig records them.
- **`--hardware` is refused for experiments other than `malus_law`**, because the rig is the polariser.
- **Rate-limit default raised from 5 to 10 moves/s.** The firmware takes at least 0.12 s per move and reading, so legitimate traffic on the real rig can reach about 7–8 moves/s, and 5/s would have rejected normal runs. The red-team results did not change: the moves-without-reading interlock still stops every burst first.
- **Docs:** `docs/wokwi.md` (setup, run, manual check, real rig, troubleshooting). The README's and the old PDF generator's firmware claims were corrected.

### Why
The hardware part of the project needs something real to show: the actual firmware on a simulated Uno, a real wiring diagram, and the unchanged Python driver talking to it. The firmware is also a second, independent safety layer, so a host bug alone cannot drive the servo outside its range.

### How it was verified
- **`pytest -q`: 391 passed, offline, in about 114 s** (367 before). The 24 new tests in `tests/test_serial_driver.py` use a fake serial port (`tests/fake_firmware.py`) that follows the firmware's rules. They cover:
  - the Wokwi and USB labels and the bootloader wait;
  - a silent port that does not connect;
  - HOME, MOVE and READ following the emulated Malus law;
  - 9 bad commands refused with the right code, among them `MOVE abc`, `MOVE 1e5`, `MOVE 90;HOME`, `MOVE nan` and an over-long line;
  - EVENT and stray lines skipped;
  - a lost reply becoming a timeout that the guard flags;
  - a dropped link reconnected once;
  - the guard's E-stop latching the firmware (a direct MOVE is refused by the firmware too), and the operator reset clearing both;
  - the firmware watchdog;
  - **a full AutoLab run over the firmware link**: labelled firmware-in-the-loop, converged on cos² with x0 ≈ 20, one MOVE per attempt that was not rejected, report labelled, audit chain intact;
  - `--hardware` refused for non-Malus experiments;
  - the source checks: every protocol command and error code is in the `.ino`; the firmware's bounds and emulated optics equal the digital twin's Malus spec; `wokwi.toml` points to the built files; `diagram.json` wires D9 and A0.
- **All three sensor builds compile** with arduino-cli 1.5.2 (arduino:avr 1.8.8) with `--warnings all` and no warnings from the sketch:

  | Build | Flash | RAM |
  |---|---|---|
  | EMULATED | 9,310 B (28%) | 417 B (20%) |
  | LDR | 9,088 B (28%) | 417 B |
  | TSL2591 | 12,772 B (39%) | 696 B (33%) |

  A build with two sensors set fails on the `#error`, as intended. The toolchain was set up in a temporary folder; nothing was installed on the system.
- **Red team rerun with the new rate limit:** identical to Phase 4 (59 cases, 0 unsafe commands; no change to `results/redteam.csv`).

### Known limitations
- **Not yet run in Wokwi.** That needs the VS Code extension and its licence; it is your step, see docs/HANDOFF.md. The full-run test uses the fake port, which follows the firmware's logic but is not the compiled code.
- **SENSOR_LDR's lux conversion is nominal** (GL5528, R10 = 50 kΩ, γ = 0.7) until it is calibrated against a lux meter.
- **The emulated sensor's noise uses a fixed seed (12345).** Firmware runs are not tied to the AutoLab run seed, so a Wokwi run is not reproducible from `--seed`. It is repeatable from a fresh board start with the same commands.
- **The servo has no position feedback,** so on the real rig a stuck servo can only show up through its readings.

## Phase 7: Dashboard (2026-10-22)

### What changed
- **Streamlit dashboard** (`streamlit run app/Home.py`), multipage:
  - **Home:** what AutoLab is, the three data-source labels, and the session's access status.
  - **Live Lab.** The sidebar sets the research question, experiment (the operator's view; agents never see the name), blind mode, agents (mock, no LLM, or a real LLM when allowed, with a model id), non-idealities, faults, seed and budget.
    - Run starts the experiment in a background thread. The page redraws from the run's events about three times a second.
    - The chart shows the measurements, the GP mean ± 2σ, the best model's curve and rejected or flagged requests. Below it are the candidate-model table (ΔBIC, and "not counted: parsimony") and the stopping criteria.
    - Alongside: rig status, a feed of agent reasoning (Planner, Analyst, Executor, Safety guard, System; each message says who wrote it, LLM model, mock or rule) and a safety-event log. At the end: the result, its equation and the Writer's discussion.
    - **E-STOP** latches the running engine's guard (`DiscoveryEngine.emergency_stop`); the run aborts cleanly as a safety abort.
    - A prominent data-source badge sits at the top.
  - **Run History:** every record in `runs/` and `runs/demo/`, newest first.
    - Summary metrics, including whether the audit chain is intact.
    - A replay slider over the attempts: the chart, the fits and the reasoning as they were at that attempt.
    - Every attempt and every fault in tables.
    - Downloads of the record (JSON), the attempts (CSV) and the report (Markdown).
  - **Benchmark:** the figures (with PDF downloads), a filterable `benchmark_summary.csv`, the red-team summary and table, the Phase 3 fault sweep, and `RESULTS.md`. It reads `results/` only, and shows the mock-provider banner prominently.
  - **System:** an architecture diagram (graphviz), the data flow, the safety layers and feature table and the standards mapping (from `docs/safety_and_standards.md`), the firmware summary, the Wokwi wiring, and a place for the circuit screenshot (`docs/img/wokwi_circuit.png`).
- **Logic is kept out of the page files** (`app/lib/`):
  - `access.py`: the public-demo policy and hosted secrets;
  - `live.py`: the run thread, event folding and E-STOP;
  - `views.py`: tables, curves, replay and the record index;
  - `charts.py`: the Altair chart;
  - `style.py`: page setup, badge and cards; agent text is HTML-escaped.

  The pages only call the engine and read results.
- **Public-demo protections.**
  - A real LLM is offered only when an API key is configured AND `APP_PASSWORD` is set AND the visitor has entered it (constant-time compare). Otherwise every run uses the mock provider and the page says why.
  - Per session: at most 10 runs, of which at most 3 with a real LLM. Budgets are capped at 30 measurements (mock or no LLM) and 20 (real LLM).
  - Hosted Streamlit secrets fill the environment where `.env` has nothing; secrets are never displayed.
- **Look:** a light theme (`.streamlit/config.toml`) using the benchmark palette, without emoji. The chart uses the validated categorical colours with a distinct mark per series.
- **Tests run the pages headless** with Streamlit's `AppTest`.
- `streamlit` was added to `requirements.txt`.

### Why
An examiner should be able to watch AutoLab work (what each agent proposed, what the guard refused, how the evidence built up until the rule stopped) and then open any run afterwards. The results chapter's figures should be one click away, and none of this should put the API credit at risk on a public link.

### How it was verified
- **`pytest -q`: 404 passed, offline, in about 120 s** (391 before). The 13 new tests in `tests/test_dashboard.py` cover:
  - the access matrix: no key, no password, wrong password, right password, the session cap, and `provider_for` never returning litellm without access;
  - hosted secrets filling the environment without overriding it;
  - **a live run streaming step by step**: the page's snapshots saw at least 4 different point counts, rising, with Planner, Analyst and System messages, the GP band, the fits and the stop;
  - **E-STOP mid-run**: a safety abort, "E-stop latched (operator pressed E-STOP on the dashboard)", the trip in the audit log;
  - the tables, curve, replay and record index from a real record;
  - **all five pages rendered with AppTest without an exception**;
  - Live Lab run from the page (badge, result "Cosine-squared with phase", metrics, record saved), offering a real LLM only with a key and the right password (never chosen, so no call);
  - Run History with the replay slider moved and all three downloads present.
- **Done when (local run).** `streamlit run app/Home.py --server.port 8599 --server.headless true` started and `/_stcore/health` answered `ok`, then it was stopped. I could not look at the page in a browser; the streaming and AppTest checks above stand in for that.

### Known limitations
- **The Live Lab drives the digital twin only.** The firmware and the rig are run from the CLI (`--hardware`), because a serial port on a hosted app makes no sense and is not safe on a public link.
- **The redraw is a polling loop (about 0.35 s).** Each redraw rebuilds the chart; with many points this could flicker on a slow machine. Not tested in a browser.
- **Session caps live in the Streamlit session.** A visitor who reloads the page starts a new session, so the caps limit accidents, not determined abuse. The password is what protects the API credit.
- **Run History lists only top-level `runs/` and `runs/demo/`,** not sweep or benchmark subfolders, and at most 200 records.

## Phase 5.1: Free-tier real-LLM subset and the llm.check fix (2026-10-22)

### What changed
- **`python -m llm.check` now pings each model the way the agents ask it.**
  - The old ping capped the answer at 32 output tokens and allowed no re-ask. A reasoning model (gpt-oss on Groq) spends 32 tokens on reasoning and returns nothing. Groq's JSON mode then rejects the empty answer with `json_validate_failed`, so the check said "failed" for models that work in real runs.
  - The ping now goes through `LLMClient.complete_json` with the agents' settings: the same output-token cap, the same JSON mode (schema where litellm says the model supports it, never `strict`), timeout, temperature and JSON re-asks.
  - Three differences remain: the chain is the one model being checked, the cache is off, and a rate limit is reported at once instead of waited out.
- **`json_validate_failed` is handled like an invalid answer** everywhere, not only in the check: the client re-asks the same model with the reason fed back (up to `max_json_retries`), then moves to the next model. Before, it counted as a generic 400 error and the model was dropped at once.
- **Free-tier pacing in the LLM client** (`LLMConfig`):
  - `min_request_interval_s` (default 0, off) spaces network requests in the process.
  - `rate_limit_cooldown_s` (default 300 s): a model still rate-limited after its retries is skipped for 5 minutes and the next model is asked directly, instead of being waited out on every call. A daily quota will not come back within minutes.
- **New condition `hidden_apparatus`** (`AgentConfig.describe_apparatus=False`, CLI `python main.py --hide-apparatus`): the agents get the variable names, units and bounds, but the apparatus description is replaced by a neutral sentence. Unlike blind mode, the names stay. The physics is the clean twin. The report says when the apparatus was hidden.
- **`python -m benchmark.run --subset`**, the free-tier replacement for the paid full real-LLM benchmark:
  - malus_law, inverse_square_law, rc_charging and diode_iv; conditions clean, faults and hidden_apparatus; llm_only and autolab with the real provider (default `litellm`); 3 seeds. That is 72 real-LLM runs.
  - The 480 non-LLM baseline rows (grid, random, gp_only; 20 seeds) are reused from the full benchmark's store (`runs/benchmark/full/rows`). They are not rerun. For hidden_apparatus, the baselines' clean rows are shown again and marked `reused_from = clean`, because a method without an LLM never reads the description.
  - No learning curves and no equal-budget stage.
  - **Pacing:** real-LLM runs go one at a time in the main process, at most `--rpm` requests per minute (default 10). Each run prints a progress line naming the models that answered.
  - **Resume:** every finished run is stored at once. A run cut short because no model was available is *not* stored, so it is retried next time. After 2 such runs in a row (a daily limit, most likely), the benchmark stops with exit code 3 and says how many runs remain. Ctrl-C exits with 130. Rerunning the same command continues, and cached answers replay for free.
  - **Rows are keyed by the model chain** (a fingerprint of LLM_MODEL + LLM_FALLBACKS), so answers from different models are never mixed in one table.
  - **`--dry-run`** prints the chain, the runs to do, the estimated calls and input/output tokens (from the mock benchmark's rows where they exist), and the hours at the chosen pace.
  - **Labels:** RESULTS.md and every figure footer say "Reduced subset, real LLM: <model (calls), …>", built from the models that actually answered. A mock rehearsal says "Reduced subset, mock provider (a rehearsal: not a real LLM)".
  - RESULTS.md gets a "same seeds only" table, because the LLM methods ran 3 seeds and the baselines 20.
  - Output goes to `results/subset/`.
- Benchmark plumbing:
  - The CSV writer takes the union of all rows' columns.
  - The run count in RESULTS.md excludes reused rows.
  - The figure footer is built by one function (`footer_text`).

### Why
The full real-LLM benchmark needs about 6,700 calls. That needs paid Claude credit, and the free tiers (Gemini Flash-Lite, then Groq) cannot cover it in reasonable time. The subset asks the questions that matter most for the report with about 1,100 calls:
- Does a real LLM change anything over GP only on four representative laws?
- Does it help or hurt under faults?
- How much of its contribution comes from being told what the apparatus is?

The full benchmark stays available for when credit exists.

### How it was verified
- **`pytest -q`: 416 passed, offline, in about 2 minutes** (404 before). New tests:
  - `tests/test_llm_client.py`: a mocked Groq `BadRequestError` with `json_validate_failed` and an empty `failed_generation`.
    - The ping now succeeds because it gets the agents' token cap (the mocked model fails below 256 tokens, as gpt-oss did).
    - A one-off `json_validate_failed` is re-asked and the call succeeds.
    - `llm.check` shows the reason when it persists.
  - Pacing across two clients.
  - A rate-limited model being skipped while it cools off.
  - `tests/test_benchmark.py`:
    - the subset plan: 72 LLM runs, 480 baselines, no hidden_apparatus for non-LLM methods, the chain fingerprint in the keys, no stage 2;
    - make_config for hidden_apparatus: clean physics, no description, 6 s spacing at 10 RPM, mock never paced;
    - the planner's actual prompt: the neutral sentence, names and units present, the real description absent;
    - an unavailable run not stored, 2 in a row stopping the benchmark, the rerun doing only the missing runs;
    - exit code 3;
    - the dry-run text;
    - the label and figure footer naming the models;
    - an end-to-end mock rehearsal: a reused baseline row shown under hidden_apparatus but not run again, and the RESULTS.md banner.
- **Dry run with the real `.env` chain** (no requests made): gemini/gemini-3.5-flash-lite → groq/openai/gpt-oss-120b → groq/openai/gpt-oss-20b. 72 runs, about 1,100 calls, ~670k input and ~60k visible output tokens, about 1.8 h at 10 requests per minute.
- **A mock rehearsal of the whole subset** (`--subset --provider mock`) wrote RESULTS.md, the CSVs and figures with the rehearsal label, into the scratchpad, not `results/`.

### Known limitations
- **No real LLM was called.** The Groq fix is verified against a mocked error copied from the reported message, not against Groq. `results/subset/` does not exist yet: the subset is run by Nikhil (see docs/HANDOFF.md).
- **The free-tier limits are not verified.** 10 requests per minute is a conservative guess for Gemini Flash-Lite and Groq. Daily limits change and differ by account; the benchmark stops cleanly when they are reached rather than predicting them.
- **The token estimate excludes hidden reasoning tokens.** gpt-oss on Groq reasons before answering, which counts against Groq's token-per-minute limit.
- **3 seeds give wide intervals.** Differences between LLM methods and baselines mix the method with the seed set (3 vs 20); the "same seeds only" table and the figure footers say so.
- **The answering model can change during a run** when Gemini runs out and Groq takes over. The label lists every model with its share of calls, and each call in the run records names its model, but a table row is not tied to a single model.

### Fix after Phase 5.1: a clear stop when the rig or Wokwi is not reachable
- **What changed.** If the serial rig did not connect (for example Wokwi not started), the failure was logged and the run went on to HOME, which crashed with a pyserial traceback. Now the run stops before anything moves with `RigConnectionError` ("could not connect to the rig on <port> (<reason>)"). The run record keeps the error. The CLI prints a hint: for `rfc2217://` to start the Wokwi simulator and keep its tab visible, for USB to check the cable, firmware and port.
- `docs/wokwi.md` now says plainly that the firmware is already built (no arduino-cli needed) and lists the common setup errors.
- **Verified:** a new test refuses the connection the way pyserial does when nothing listens on port 4000 and checks the message, the hint and the record; by hand, the real command without a simulator now prints the hint. `pytest -q`: 417 passed.
