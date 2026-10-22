# AutoLab — project context for Claude Code

AutoLab is a final-year capstone project (B.Tech, Dept. of Electrical & Electronics Engineering, MIT-WPU Pune, AY 2026-27; team: Nikhil Kumar, Piyush Warghane). It is a benchtop **self-driving laboratory**: LLM agents plan experiments, a Gaussian-Process active-learning engine picks the next measurement, a hard-coded safety guard sits between the agents and the actuator, and the system fits candidate physical laws to the data and writes a report.

The work is assessed by examiners (including an industry expert) and will become a research paper. **Scientific honesty matters more than impressive numbers**: every result must be reproducible from a seed and labelled with where it came from (digital twin, firmware-in-the-loop, or real hardware).

The full build plan is in `ROADMAP.md`. Work on one phase at a time.

## Architecture (target)

```
Streamlit dashboard (app/)          CLI (main.py)          Benchmark harness (benchmark/)
              \                         |                         /
               +----------- DiscoveryEngine (agents/orchestrator.py) ----------+
               |   emits events -> callback (dashboard / console / run log)    |
               |                                                               |
  LLM layer (llm/)       Agents (agents/)             Optimizer (optimization/)
  litellm | mock | none  Planner, Analyst, Writer     GP active learning
               |                                                               |
               +---------------- HardwareSafetyGuard (hardware/safety_guard.py)
                                   |                         |
                     PhysicsSimulationRig            SerialHardwareRig
                     (digital twin, faults)          (Arduino over USB or Wokwi rfc2217://)
```

Data flow per iteration: Planner/Analyst propose a position (JSON) → SafetyGuard validates → rig moves and reads → Analyst fits models → GP proposes the next point → stopping rule decides.

## Hard rules

1. **The LLM never talks to hardware directly.** It only returns structured JSON (validated with Pydantic). Every position goes through `HardwareSafetyGuard`. A rejected command is fed back to the agent as an observation, never retried blindly.
2. **Never `eval`/`exec` LLM output, and never pass it to `sympy.parse_expr` or `sympify`** (both call `eval` internally). Model expressions proposed by an LLM are parsed with `ast.parse(mode="eval")` and checked node by node against a whitelist: `x`, the declared parameter names, numbers, `pi`, `e`, `+ - * / **`, unary minus, and one-argument calls to `sin cos tan exp log sqrt abs radians`. Attributes, subscripts, strings, keywords, lambdas, `__`, undeclared names and anything over 200 characters are rejected. The validated tree is evaluated by walking it with numpy (`analysis/candidate_models.py`).
3. **Secrets live only in `.env`** (local) or Streamlit secrets (cloud). Never hard-code or print API keys. `.env` is git-ignored.
4. **Three LLM providers, selectable by `LLM_PROVIDER`:** `litellm` (real model: Claude `LLM_MODEL` first, then Gemini models in `LLM_FALLBACKS`; record which model answered; never fall back silently to mock in a real run), `mock` (deterministic, no network; used by all tests and the public demo), `none` (no LLM: heuristic fallback; used as the ablation baseline). Everything must run with `mock` and no API key.
5. **API budget is tight** (paid Claude credit, small Gemini free quota): keep prompts compact, cache responses, back off on 429, never loop on retries.
   The LLM must never set safety bounds or the simulator's ground truth (a mistake found in an earlier teammate version).
6. **Tests never hit the network or a real LLM.** Use the `mock` provider. `pytest -q` must pass before every commit.
7. **Reproducibility:** every run takes a `seed`; every random draw uses a `numpy.random.Generator` created from it (no global `np.random.seed`). Every run is saved as a JSON record in `runs/` (config, seed, every step, every LLM call with tokens/cost/latency, fits, final result).
8. **Blind analysis:** the Analyst must not be told the true law or the experiment's internal name. It sees only variable names, units and data.
9. **Label results honestly.** Dashboard, plots and reports must show the data source: `DIGITAL TWIN`, `FIRMWARE-IN-THE-LOOP (Wokwi)` or `PHYSICAL RIG`.
10. Keep the hardware abstraction: anything that works on the simulator must work on `SerialHardwareRig` without changes to agents or optimizer.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                               # all tests (mock LLM)
python main.py --experiment malus_law                   # one run, CLI
python main.py --experiment malus_law --provider mock   # no API key needed
python main.py --experiment diode_iv --faults default --nonideal   # digital twin with faults and non-idealities
python -m scripts.twin_sweep                            # seeded fault/non-ideality sweep -> results/
python -m safety.redteam                                # red-team suite -> results/redteam.csv (0 unsafe commands)
streamlit run app/Home.py                               # dashboard (after Phase 7)
python -m benchmark.run --quick                         # benchmark (after Phase 5)
```

## Conventions

- Python 3.11+; type hints everywhere; Pydantic v2 models for all data crossing module boundaries.
- Use `datetime.now(timezone.utc)`, not `datetime.utcnow()`.
- Use `logging`, not `print`, in library code. The `rich` console output belongs only to the CLI layer.
- Small, focused modules; one responsibility each. Prefer editing existing files over creating parallel versions.
- Each new feature comes with tests. Each phase ends with the **Phase checkpoint** from ROADMAP.md (tests, CHANGELOG, commit `Phase N: ...`, tag `phase-N`, push). The `docs/CHANGELOG.md` entry is written as plain notes (what changed, why, how it was verified). These notes feed the project report.
- Commit after each working step with a clear message; tag `phase-N` at the end of each phase.
- `reference/` holds a teammate's older version for read-only inspiration. It is git-ignored; never import from it.

## Known issues in the original code (fixed during the roadmap)

- No LLM was called anywhere: `LeadScientistAgent` used keyword matching; `litellm` was unused.
- The Analyst was told `experiment_type`, so it only tried the "right" law. It was circular.
- Convergence was forced at exactly 8 points (`n >= 8`) because the R² threshold was always met.
- `exponential_model` existed but was never fitted, so Beer-Lambert data was mis-identified as quadratic.
- Expected Improvement maximises the response. It is not suitable for learning a whole curve; use uncertainty sampling or integrated variance reduction.
- The firmware reads an analog LDR on A0, while the docs claim TSL2591; there is no stepper code. Docs and firmware must agree.
- The safety limit [0,180] was reused for the distance experiment.
