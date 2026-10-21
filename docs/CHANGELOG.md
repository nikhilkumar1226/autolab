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
