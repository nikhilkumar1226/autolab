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
2. **Never `eval`/`exec` LLM output.** Model expressions proposed by an LLM are parsed with `sympy.parse_expr` using a whitelist of symbols and functions only.
3. **Secrets live only in `.env`** (local) or Streamlit secrets (cloud). Never hard-code or print API keys. `.env` is git-ignored.
4. **Three LLM providers, selectable by `LLM_PROVIDER`:** `litellm` (real model, name in `LLM_MODEL`, fallbacks in `LLM_FALLBACKS`; free tiers only: Groq main, Gemini backup, local Ollama), `mock` (deterministic, no network; used by all tests and the public demo), `none` (no LLM: heuristic fallback; used as the ablation baseline). Everything must run with `mock` and no API key.
5. **Free-tier budget is tight** (Groq ~200k tokens/day per model): keep prompts compact, cache responses, back off on 429, never loop on retries.
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
streamlit run app/Home.py                               # dashboard (after Phase 7)
python -m benchmark.run --quick                         # benchmark (after Phase 5)
```

## Conventions

- Python 3.11+; type hints everywhere; Pydantic v2 models for all data crossing module boundaries.
- Use `datetime.now(timezone.utc)`, not `datetime.utcnow()`.
- Use `logging`, not `print`, in library code. The `rich` console output belongs only to the CLI layer.
- Small, focused modules; one responsibility each. Prefer editing existing files over creating parallel versions.
- Each new feature comes with tests. Each phase ends with a short `docs/CHANGELOG.md` entry, written as plain notes (what changed, why, how it was verified). These notes feed the project report.
- Commit after each working step with a clear message.

## Known issues in the original code (fixed during the roadmap)

- No LLM was called anywhere: `LeadScientistAgent` used keyword matching; `litellm` was unused.
- The Analyst was told `experiment_type`, so it only tried the "right" law. It was circular.
- Convergence was forced at exactly 8 points (`n >= 8`) because the R² threshold was always met.
- `exponential_model` existed but was never fitted, so Beer-Lambert data was mis-identified as quadratic.
- Expected Improvement maximises the response. It is not suitable for learning a whole curve; use uncertainty sampling or integrated variance reduction.
- The firmware reads an analog LDR on A0, while the docs claim TSL2591; there is no stepper code. Docs and firmware must agree.
- The safety limit [0,180] was reused for the distance experiment.
