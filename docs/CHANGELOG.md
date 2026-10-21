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
- **Inverse-square runs can stop confidently on the wrong law.** In the sweep, 6 of 20 runs converged on the wrong law:
  - exponential decay for seeds 3, 4, 7 and 17;
  - power law for seeds 10 and 16.

  Four of these stopped at the 5-point minimum. The cause is not a fitting failure: the inverse-square fit reached the brute-force global optimum in every case. With 5 points, a 3-parameter model has only 2 residual degrees of freedom. A wrong model can then pass almost exactly through the noisy points and win a large ΔBIC by chance (up to 18.4 here).
- **Raising `min_points` would fix most of this, but it is not the specified default.** The same sweep with `StoppingConfig(min_points=8)` gives 19/20 correct for inverse-square, with one power-law stop (seed 10). Malus and Beer-Lambert stay at 20/20, but stop at 8 points instead of 5–7. `min_points = 10` gives the same accuracy. The default stays at 5 as specified for Phase 1. Whether to raise it is a decision for the next review.
- **Malus always stops at the 5-point minimum.** At σ = 10 against a 1000 lux amplitude, ΔBIC is already about 30 at n = 5, so this floor reflects strong evidence, not a forced stop.
- **The cos² model assumes x is in degrees** (period 180 units).
- **Offsets with large absolute error can still pass.** Span-normalised errors can accept an offset that is poorly known in absolute terms (Beer-Lambert seed 1: C = 7.0 ± 13.3 lux, true value 15). The raw SE is always reported next to it.
