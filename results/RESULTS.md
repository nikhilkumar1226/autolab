# AutoLab benchmark results

*Generated 2026-10-22 06:09 UTC by `python -m benchmark.run --full --provider mock`. Every number below comes from `benchmark_runs.csv` in this folder.*

> **Data source: DIGITAL TWIN** for every run.
>
> **LLM methods ran with the mock provider**, a deterministic stand-in, not a language model. In AutoLab-mock the Planner always hands the choice to the GP, so *AutoLab equals GP only by construction*. LLM-only-mock uses a fixed rule: the widest gap, or where the two best models disagree. These rows show the pipeline works end to end; **they say nothing about what a real LLM adds**. That needs `python -m benchmark.run --full --provider litellm`.
>
> Check: on the 120 adaptive runs where both have the same experiment, condition and seed, AutoLab-mock and GP only took identical measurements and found the same law in **120 of 120**. **Any difference between them in the tables below comes only from the seed sets** (AutoLab uses seeds 0-4, GP only 0-19), not from the method. The same applies to comparisons of LLM-only (5 seeds) with the baselines (20 seeds).

Seeds: 20 for grid / random / GP only, 5 for LLM-only and AutoLab. Runs: 3680 (0 ended in an error). Experiments: malus_law, inverse_square_law, beer_lambert_law, rc_charging, diode_iv, ldr_response, hookes_law, stefan_boltzmann. Every method starts from the same 4-point design and uses the same stopping rule and safety guard.

## 1. Law identified correctly (blind), all experiments pooled

| Method | Clean | Non-idealities on | Faults injected |
|---|---|---|---|
| AutoLab (full) | 100% [91%, 100%] (n=40) | 98% [87%, 100%] (n=40) | 95% [83%, 99%] (n=40) |
| LLM only | 100% [91%, 100%] (n=40) | 98% [87%, 100%] (n=40) | 100% [91%, 100%] (n=40) |
| GP only | 99% [96%, 100%] (n=160) | 99% [96%, 100%] (n=160) | 94% [89%, 97%] (n=160) |
| Random | 98% [95%, 99%] (n=160) | 96% [91%, 98%] (n=160) | 94% [89%, 97%] (n=160) |
| Grid | 99% [97%, 100%] (n=160) | 98% [95%, 99%] (n=160) | 97% [93%, 99%] (n=160) |

Wilson 95% intervals in brackets.

## 2. Law identified, per experiment

**Clean**

| Experiment | AutoLab (full) | LLM only | GP only | Random | Grid |
|---|---|---|---|---|---|
| malus_law | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| inverse_square_law | 5/5 | 5/5 | 19/20 | 19/20 | 19/20 |
| beer_lambert_law | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| rc_charging | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| diode_iv | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| ldr_response | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| hookes_law | 5/5 | 5/5 | 19/20 | 18/20 | 20/20 |
| stefan_boltzmann | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |

**Non-idealities on**

| Experiment | AutoLab (full) | LLM only | GP only | Random | Grid |
|---|---|---|---|---|---|
| malus_law | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| inverse_square_law | 4/5 | 4/5 | 19/20 | 18/20 | 19/20 |
| beer_lambert_law | 5/5 | 5/5 | 20/20 | 19/20 | 19/20 |
| rc_charging | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| diode_iv | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| ldr_response | 5/5 | 5/5 | 20/20 | 19/20 | 20/20 |
| hookes_law | 5/5 | 5/5 | 19/20 | 17/20 | 19/20 |
| stefan_boltzmann | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |

**Faults injected**

| Experiment | AutoLab (full) | LLM only | GP only | Random | Grid |
|---|---|---|---|---|---|
| malus_law | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| inverse_square_law | 4/5 | 5/5 | 14/20 | 13/20 | 17/20 |
| beer_lambert_law | 5/5 | 5/5 | 20/20 | 18/20 | 20/20 |
| rc_charging | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| diode_iv | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| ldr_response | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |
| hookes_law | 4/5 | 5/5 | 16/20 | 19/20 | 18/20 |
| stefan_boltzmann | 5/5 | 5/5 | 20/20 | 20/20 | 20/20 |

## 3. Measurements to stop and parameter error at stop

Parameter error: the largest error over the true law's parameters (normalised like the stopping rule: by |true value| for a scale, by the x-span for a location, by the response span for an offset), in %, for the fit of the true law at the end of the run.

| Method | Condition | Measurements, median | mean [95% CI] | Parameter error, median % | Stopped by budget |
|---|---|---|---|---|---|
| AutoLab (full) | Clean | 8 | 9.3 [8.2, 10.4] | 0.79 | 0/40 |
| AutoLab (full) | Non-idealities on | 8 | 12.9 [10.3, 15.5] | 1.36 | 6/40 |
| AutoLab (full) | Faults injected | 8 | 9.9 [8.2, 11.6] | 1.43 | 1/40 |
| LLM only | Clean | 8 | 8.8 [8.1, 9.6] | 0.95 | 0/40 |
| LLM only | Non-idealities on | 8 | 12.6 [9.9, 15.3] | 1.13 | 6/40 |
| LLM only | Faults injected | 8 | 9.7 [8.2, 11.1] | 1.05 | 1/40 |
| GP only | Clean | 8 | 10.2 [9.4, 11.0] | 1.14 | 3/160 |
| GP only | Non-idealities on | 8 | 12.7 [11.5, 13.9] | 1.81 | 20/160 |
| GP only | Faults injected | 8 | 11.7 [10.5, 12.8] | 1.46 | 11/160 |
| Random | Clean | 8 | 11.0 [10.0, 12.0] | 1.18 | 10/160 |
| Random | Non-idealities on | 8 | 13.9 [12.5, 15.3] | 2.01 | 30/160 |
| Random | Faults injected | 8 | 12.7 [11.4, 14.0] | 1.51 | 16/160 |
| Grid | Clean | 8 | 10.6 [9.7, 11.5] | 0.96 | 4/160 |
| Grid | Non-idealities on | 8 | 12.3 [11.1, 13.5] | 2.11 | 17/160 |
| Grid | Faults injected | 8 | 12.0 [10.7, 13.3] | 1.02 | 23/160 |

## 4. Equal budget: baselines at AutoLab's median number of measurements

Grid, random and GP-only were run to exactly AutoLab's median n for each experiment and condition. AutoLab's own adaptive runs are shown for reference.

| Method | Clean | Non-idealities on | Faults injected |
|---|---|---|---|
| GP only | 94% [90%, 97%]; error 1.03% | 96% [91%, 98%]; error 2.66% | 90% [84%, 94%]; error 1.42% |
| Random | 94% [90%, 97%]; error 1.29% | 93% [88%, 96%]; error 2.27% | 89% [84%, 93%]; error 1.51% |
| Grid | 95% [90%, 97%]; error 0.99% | 96% [92%, 98%]; error 2.10% | 83% [77%, 88%]; error 0.90% |
| AutoLab (full) | 100% [91%, 100%]; error 0.79% | 98% [87%, 100%]; error 1.36% | 95% [83%, 99%]; error 1.43% |

Budgets used: beer_lambert_law clean 8; beer_lambert_law faults 8; beer_lambert_law nonideal 8; diode_iv clean 9; diode_iv faults 9; diode_iv nonideal 19; hookes_law clean 8; hookes_law faults 8; hookes_law nonideal 8; inverse_square_law clean 14; inverse_square_law faults 15; inverse_square_law nonideal 30; ldr_response clean 8; ldr_response faults 8; ldr_response nonideal 8; malus_law clean 8; malus_law faults 8; malus_law nonideal 8; rc_charging clean 8; rc_charging faults 8; rc_charging nonideal 8; stefan_boltzmann clean 8; stefan_boltzmann faults 8; stefan_boltzmann nonideal 8.

## 5. Fault detection and recovery (faults condition)

| Method | Faults reached | Recovered | Spike recovered | Stuck actuator recovered | Dropout recovered | False alarms per run |
|---|---|---|---|---|---|---|
| AutoLab (full) | 120 | 98% [94%, 100%] | 38/40 | 40/40 | 40/40 | 0.00 |
| LLM only | 120 | 99% [95%, 100%] | 39/40 | 40/40 | 40/40 | 0.00 |
| GP only | 480 | 96% [94%, 98%] | 143/160 | 160/160 | 160/160 | 0.00 |
| Random | 480 | 97% [95%, 98%] | 144/160 | 160/160 | 160/160 | 0.04 |
| Grid | 480 | 96% [93%, 97%] | 139/160 | 160/160 | 160/160 | 0.00 |

Random outliers (non-idealities on, all methods): 31 of 230 detected.

## 6. LLM use and cost per run (adaptive runs, all conditions)

The mock provider costs nothing. *Estimated* cost of a real model: prompt and answer characters / 4 as tokens, priced with litellm's bundled price map for `anthropic/claude-haiku-5-5` ($0.10 / $0.50 per million input / output tokens). ₹ at 88 INR/USD (assumed rate).

| Method | LLM calls | Tokens in / out | Cost, USD | Cost, INR | Wall time, s |
|---|---|---|---|---|---|
| AutoLab (full) | 16.7 | ~10392 / ~933 (est.) | ~0.0015 (est.) | ~0.132 (est.) | 2.59 |
| LLM only | 16.1 | ~9812 / ~888 (est.) | ~0.0014 (est.) | ~0.125 (est.) | 1.89 |
| GP only | 0.0 | ~0 / ~0 (est.) | ~0.0000 (est.) | ~0.000 (est.) | 2.53 |
| Random | 0.0 | ~0 / ~0 (est.) | ~0.0000 (est.) | ~0.000 (est.) | 2.52 |
| Grid | 0.0 | ~0 / ~0 (est.) | ~0.0000 (est.) | ~0.000 (est.) | 1.82 |

## Figures

- `figures/fig_param_error_vs_n.png` (and `.pdf`): Largest parameter error of the true law's fit after each measurement (geometric mean and 95% CI over seeds, clean twin, runs to 20 points): how fast each method pins down the parameters.
- `figures/fig_law_accuracy.png` (and `.pdf`): Share of runs that identified the right law blind, per method and condition (all experiments pooled, Wilson 95% CI).
- `figures/fig_equal_budget_accuracy.png` (and `.pdf`): The same, with grid / random / GP-only run to exactly AutoLab's median number of measurements, so accuracy is compared at equal cost.
- `figures/fig_fault_recovery.png` (and `.pdf`): What became of each injected fault (recovered, missed, not reached), per method, in the faults condition.
- `figures/fig_cost_per_run.png` (and `.pdf`): LLM calls and cost per run, in USD with INR alongside (cost estimated for a real model from prompt sizes).

## Files

- `benchmark_runs.csv`: one row per run (plan, outcome, parameter errors, faults, LLM usage, cost, wall time, run id).
- `benchmark_summary.csv`: mean, std and 95% CI per run kind × method × experiment × condition, and pooled over experiments (`ALL`).
- Run records: `runs/benchmark/<mode>/records/` (git-ignored); every row's `run_id` names its record.
