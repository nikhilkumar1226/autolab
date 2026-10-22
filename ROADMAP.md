# AutoLab Roadmap

## Final goal

A **deployed web dashboard** where anyone can type a research question and watch AutoLab run an experiment live. Real LLM agents plan, the Gaussian Process picks each measurement, the safety guard blocks anything unsafe, and the Analyst identifies the law **blind**. The system runs on a realistic digital twin, and the same code drives the real Arduino firmware in a simulator (Wokwi).

Behind it sits a **benchmark** that shows, with repeated seeded runs:
1. active learning vs. grid and random sampling,
2. what the LLM adds (full system vs. no-LLM vs. LLM-only),
3. the safety guard blocking 100% of unsafe commands, including deliberate attacks,
4. recovery from injected faults.

Every number and figure is exported to `results/`. Those files are the source for the black book, the poster and the research paper.

## How to work (read once)

- **One phase = one fresh Claude Code conversation.** Start a new chat for each phase so the context stays clean. Claude Code reads `CLAUDE.md` automatically.
- **Plan first.** For every phase, switch the prompt box to **Plan mode**, paste the prompt, read the plan, ask questions, then approve. Don't approve a plan you don't understand. Ask it to explain instead.
- **Check, don't trust.** A phase is done only when *you* have run `pytest -q` and the "Done when" checks yourself.
- **Commit often.** Git commits are save points. If something breaks, you can go back.
- **After each phase,** push to GitHub and tell me in the claude.ai chat. I'll review the code before you start the next phase.

### Phase checkpoint (run at the end of EVERY phase)

When a phase looks finished, paste this into the same Claude Code conversation (replace N and the title):
```
Phase checkpoint for Phase N (<title>):
1. Run pytest -q and show me the result. If anything fails, fix it first.
2. Go through the "Done when" list for Phase N in ROADMAP.md one by one, run the
   commands, and tell me honestly which items pass and which don't.
3. Add a docs/CHANGELOG.md entry: what changed, why, how it was verified.
4. Make sure .env, API keys and runs/ junk are not staged (git status).
5. Commit with the message "Phase N: <title>", create the git tag phase-N, and push
   (git push && git push --tags).
6. Tick the "Done when" boxes for Phase N in ROADMAP.md and commit that too.
```
Only start the next phase after this succeeds. The tag `phase-N` is a save point: if a later phase breaks everything, you can go back to it.

---

## Phase 0 — Setup and cleanup ✅ DONE

**Goal:** a clean, version-controlled project that runs.

Do this yourself (not via Claude):
1. Delete `__MACOSX/`, every `.DS_Store`, and everything inside `output/plots` and `output/reports`.
2. Copy `CLAUDE.md`, `ROADMAP.md`, `.gitignore` and `.env.example` into the project root (the folder that contains `main.py`).
3. Copy `.env.example` to a new file called `.env`. Leave the keys empty for now and set `LLM_PROVIDER=mock`.
4. Create the virtual environment and install packages (commands in `CLAUDE.md`). Run `pytest -q`. You should see 8 passed.
5. Create a **private** GitHub repository called `autolab` and push the code. If you've never used git, ask Claude Code: *"Initialise git here, make a first commit, and walk me through pushing to a new private GitHub repo called autolab. Explain each command."*

**Done when:** the repo is on GitHub, `.env` is NOT in it, and the tests pass.

---

## Phase 1 — Foundation refactor ✅ DONE (run the check below)

**Why:** the dashboard, the benchmark and the LLM agents all need the same foundations: experiments described as data, every run recorded, live events, and no hidden hints to the Analyst.

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 1). Plan and implement Phase 1 of AutoLab:

1. Experiment registry (experiments/registry.py): an ExperimentSpec Pydantic model with
   name, description_for_agents (no hint of the law), independent variable name + units,
   dependent variable name + units, safe bounds, ground-truth function and parameters
   (used ONLY by the simulator and by benchmark scoring), and default noise settings.
   Register malus_law (deg, 0-180), inverse_square_law (cm, 5-100), beer_lambert_law
   (mg/L, 0-100). Safety bounds come from the spec, not from a global [0,180].
2. Blind Analyst: remove experiment_type from DataAnalystAgent. It always fits the full
   model library (linear, quadratic, exponential decay, cos^2 with phase, inverse-square
   with offset, power law) with sensible automatic initial guesses and bounds derived
   from the data, and ranks by BIC. Report delta-BIC to the runner-up.
3. Proper stopping rule (replace the hard n>=8): stop when the best model beats the
   runner-up by delta-BIC >= 6 AND the relative standard error of every fitted parameter
   is below a configurable threshold (default 5%) AND at least 5 points exist; also stop
   at a max-measurement budget. Record which criterion stopped the run.
4. Reproducibility: every run takes a seed; use numpy Generator objects passed down
   explicitly (simulator noise, random baselines). No global np.random.seed.
5. Run records: save every run as JSON in runs/<run_id>.json (config, seed, experiment,
   data source label, every step, fits per iteration, stop reason, timings).
   Add a loader.
6. Event callback: DiscoveryEngine accepts an on_event(event) callback. Emit typed events
   (run_started, hypothesis, measurement, safety_rejection, analysis, gp_update,
   stopped, report_ready). Move all rich console printing into a console callback used
   only by main.py.
7. Fix datetime.utcnow deprecations. Remove unused dependencies (scikit-optimize,
   tabulate) if truly unused.
8. Tests: blind analyst correctly identifies all three laws from simulated data, stopping
   rule unit tests, run record round-trip, and the same seed giving identical results.
Write a docs/CHANGELOG.md entry. Commit when pytest passes.
```

**Done when:**
- [x] `pytest -q` passes.
- [x] `python main.py --experiment beer_lambert_law` identifies **exponential decay** (it used to say quadratic).
- [x] Runs stop at different numbers of points, not always 8. (Caveat: Malus always stops at exactly 8, the new `min_points` floor; see docs/CHANGELOG.md.)
- [x] A JSON file appears in `runs/` for every run.

### Phase 1 review (done by Claude on Oct 22) and one fix before Phase 2

Review result: **Phase 1 passes.** 80/80 tests pass, Beer-Lambert is now identified as exponential, runs stop at different point counts, every run writes `runs/*.json`, and no keys are in git history. One problem, found by the team's own sweep: inverse-square runs sometimes stop early on the wrong law (14/20 correct), because 5 points leave a 3-parameter model only 2 residual degrees of freedom.

**Fix prompt** (new conversation, Plan mode):
```
Read CLAUDE.md, ROADMAP.md and docs/CHANGELOG.md (Phase 1 "Known limitations").
Fix the inverse-square false-convergence problem WITHOUT just raising the fixed floor:
1. Add a "predictive stability" criterion to analysis/stopping.py: before each new
   measurement is added, predict it with the current best model; after adding it, the
   run may converge only if (a) the same model was best in the last 2 analyses with
   delta-BIC >= 6 both times, and (b) the new point was predicted within 3 sigma
   (sigma = residual std of the fit, floored at a robust noise estimate). Keep
   min_points=5 and the budget. Make it configurable; record its value, threshold and
   pass/fail in the stop decision like the other criteria.
2. Re-run the Phase 1 seeded sweep (seeds 0-19 x 3 experiments) for three configs:
   (A) current, (B) min_points=8, (C) new stability rule. Save
   results/phase1_stopping_sweep.csv and print a table per config: correct-law count,
   median/max measurements, stop reasons.
3. Choose as default the config with >= 19/20 correct on every experiment and the
   fewest median measurements. Explain the choice with the numbers in CHANGELOG.
4. Add tests for the new criterion.
Then stop and summarise; I will run the Phase checkpoint for Phase 1.
```
Then run the **Phase checkpoint for Phase 1**. It creates the `phase-1` tag (there are no tags yet). Send me the sweep table.

---

## Piyush's version: what we take, what we don't

Piyush's copy (`AI-Laboratory`) has useful pieces. We **port the ideas** into our code instead of merging his code.

Setup (you):
1. Make a folder `reference/` inside `autolab` and put Piyush's `AI-Laboratory` folder in it.
2. **Delete `reference/AI-Laboratory/.env`** (it contains his API key) and `reference/AI-Laboratory/.git`.
3. Add the lines `reference/` and `llm_cache/` to `.gitignore`, so they never go to GitHub.

| Take (Phase 2 / 3) | Never take |
|---|---|
| `CandidateModelSpec` idea (agent-proposed candidate models) | `ground_truth_simulation_formula`: the LLM must never write the simulator's physics |
| AST-validated formula compiler (`compile_safe_model_callable`), tightened to a whitelist of numpy functions | Orchestrator code that copies `hypothesis.suggested_range` into the safety-guard limits |
| litellm JSON-mode call pattern + the mocked-LLM test style (`patch("litellm.completion")`) | `eval(...)` of formulas in `hardware/simulator.py` |
| Hooke's law and Stefan-Boltzmann as extra experiments (with fixed, hidden ground truth) | Keyword "semantic engine" as the main planner (fine only as the `none` fallback) |
| `--model` CLI flag, timezone-aware datetimes | |

---

## Phase 2 — Real LLM agents (most important phase)

**Why:** the project title says "LLM agents", but the original code never called an LLM. An examiner who asks "show me the prompt" must see a real one, along with the reasoning it produced.

**Before this phase:** put your Claude API key and the Gemini key in `.env` (see "API keys" at the end) and set up `reference/` (section above). You can build the whole phase with `LLM_PROVIDER=mock` and only switch to the real model to try it.

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 2, and the section "Piyush's version").
Plan and implement real LLM agents for AutoLab. Use reference/AI-Laboratory as READ-ONLY
inspiration for the "Take" column (CandidateModelSpec, the AST-validated compiler, the
litellm JSON call, mocked-litellm tests). Do NOT copy anything from the "Never take"
column. Never let the LLM change safety bounds or the simulator's ground truth.

1. llm/client.py: LLMClient with complete_json(system, user, schema: type[BaseModel]) that
   returns a validated Pydantic object. Providers selected by LLM_PROVIDER env var:
   - "litellm": real call via litellm using LLM_MODEL; JSON-mode / structured output;
     retry up to 2 times on invalid JSON, feeding the validation error back; timeout.
     PROVIDER PRIORITY: LLM_MODEL first (Claude, e.g. anthropic/claude-haiku-5-5), then
     the comma-separated LLM_FALLBACKS in order (Gemini: gemini/gemini-3.6-flash, then
     gemini/gemini-3.5-flash-lite). Fall back on auth errors, "credit balance too low",
     rate limits (HTTP 429, back off with jitter first), timeouts and 5xx. Never fall
     back silently to mock in a real run: if every model fails, stop the run with a clear
     error. Record which model actually answered each call. Cache identical prompts on
     disk (llm_cache/, git-ignored) so re-runs cost nothing; keep prompts compact.
     Add `python -m llm.check` that sends one tiny request to every configured model
     and prints OK / error per model (to test keys).
   - "mock": deterministic, offline, plausible responses for every agent schema.
   - "none": raises NotAvailable so agents fall back to heuristics (ablation baseline).
   Log every call into the run record: agent, prompt, raw response, parsed result,
   input/output tokens, estimated cost (litellm cost helpers), latency.
2. Planner agent (replace keyword matching): from the research question + the
   ExperimentSpec's agent-facing description, return a Hypothesis (statement, candidate
   relationships in words, independent variable, range inside the safe bounds) and an
   initial design of 3-5 points with a rationale. Positions are only proposals; the
   safety guard validates them.
3. Analyst agent: after the numerical fits, the LLM receives the fit table (BIC, delta-BIC,
   parameters, residual pattern summary) and returns: interpretation, whether residuals
   suggest a missing model, and optionally ONE new candidate model as a sympy expression in
   x with named parameters. Parse it with sympy.parse_expr using a strict whitelist
   (x, parameter symbols, + - * / **, sin, cos, exp, log, sqrt, pi) -- never eval -- then
   fit it with curve_fit and add it to the ranking. Reject and log anything else.
4. Strategy decision: each iteration the Planner may choose the next point itself or
   delegate to the GP (uncertainty sampling), and must give a reason. The numerical
   stopping rule from Phase 1 stays authoritative; the LLM writes the stop rationale.
5. Safety feedback: if the guard rejects a proposal, emit safety_rejection, give the
   rejection message back to the agent as an observation, and let it re-propose (max 2
   tries, then fall back to the GP).
6. Writer agent: writes the discussion/conclusion section of the generated report from
   the run record (no invented numbers: pass the numbers in, and tell it to use only those).
7. CLI flags: --provider {litellm,mock,none} --model <name> --seed <int> --blind.
8. Tests with the mock provider only: schemas validate, the formula whitelist rejects
   dangerous input (e.g. "__import__('os')", "np.load('x')", "().__class__"), safety
   rejection is fed back, an LLM-proposed range outside the safety bounds is clipped and
   logged (never applied to the guard), the fallback chain moves to the next model when
   the first raises an error (mock litellm), provider=none runs end to end.
When done, stop and summarise what you built; I will then run the Phase checkpoint.
```

*Change after review (Oct 22): step 3 no longer uses `sympy.parse_expr`, which calls `eval` internally. Proposed expressions are parsed with `ast.parse(mode="eval")`, checked against a whitelist and evaluated by walking the tree with numpy (CLAUDE.md rule 2).*

**Done when:**
- [ ] `python -m llm.check` shows which keys work (Claude, Gemini). (Open: the tool works and reports every model, but `.env` still has the old layout and no keys. Copy the layout from `.env.example`, add the keys and rerun it. See docs/CHANGELOG.md.)
- [x] `python main.py --provider mock` works with no key.
- [ ] With a real key, `python main.py --provider litellm` works, and the run JSON shows the real prompts, responses, tokens and cost. (Open: needs a key. Without one, the run stops with a clear error and never falls back to mock; the litellm path is tested offline with a scripted fake.)
- [x] `--provider none` still works (this is your "no LLM" baseline).

---

## Phase 3 — Realistic digital twin, new experiments, fault injection

**Why:** a perfect simulator makes any method look perfect. Real-world imperfections and faults are what make the comparison meaningful, and recovering from faults is the best argument for using an LLM.

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 3). Upgrade the digital twin (hardware/simulator.py).

1. Configurable non-idealities (all off/on via config, all seeded):
   polariser finite extinction ratio, slow ambient-light drift (random walk), sensor
   saturation (clip), ADC quantisation, actuator position error (gaussian) and backlash
   (direction-dependent offset), occasional outlier spikes.
2. New experiments in the registry, EE-flavoured, for blind discovery:
   - rc_charging: capacitor voltage vs time, V(t)=Vs(1-exp(-t/RC)), t in ms
   - diode_iv: diode current vs forward voltage (Shockley, exponential), V in volts, with
     a current-limit safety bound
   - ldr_response: LDR resistance vs illuminance (power law R = A * E^-gamma)
   - hookes_law and stefan_boltzmann (ideas from reference/AI-Laboratory), with FIXED
     ground truth in the registry, never written by an LLM
   Make sure the model library can represent each (add saturating-exponential and the
   Shockley form if needed). Agent-facing descriptions must NOT reveal the law.
3. Fault injection: a FaultSchedule in config, e.g. [{"type":"light_spike","at_step":6},
   {"type":"sensor_dropout","at_step":9,"duration":2},{"type":"stuck_actuator","at_step":7}].
   Dropout returns NaN/timeouts; stuck actuator returns the old position.
4. Detection & recovery: the safety guard flags invalid readings (NaN, saturation, sudden
   jump far outside GP prediction band, position mismatch). The Executor re-measures;
   the Analyst LLM gets the anomaly and decides (re-measure, discard, or pause). Record
   detected/recovered/missed per fault in the run record.
5. Tests for every non-ideality and fault type, and for blind identification of the 3 new
   laws with the mock/none provider.
When done, stop and summarise what you built; I will then run the Phase checkpoint.
```

**Done when:**
- [x] All 6 experiments run with faults on and off. (All 8 registry experiments, from the CLI and as a test; see docs/CHANGELOG.md, Phase 3.)
- [x] The run JSON lists each injected fault and whether it was detected and recovered. (`faults` and `fault_summary` in run-record schema 3.)

*Changes from the prompt, decided during Phase 3 (reasons and numbers in docs/CHANGELOG.md):*
- *Outliers are detected with the current best fit's prediction band (Student-t), not the GP band: the GP band gave 47 false alarms in 80 clean runs. The GP band stays available as an option.*
- *The default fault schedule is spike 6, stuck 8, dropout 10–11, because in the example the stuck actuator at cycle 7 always falls on the spike's re-measurement.*
- *Stefan–Boltzmann temperature is in kK, a parsimony rule lets a linear law converge, and a growing exponential covers the diode.*

---

## Phase 4 — Safety hardening and red-team test

**Why:** the rubric has 5 marks for quality and safety, and the paper's main claim is "hallucinations can never reach the actuator". This phase proves it.

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 4). Harden and prove the safety layer.

1. HardwareSafetyGuard: reject NaN/inf/non-numeric, enforce per-experiment bounds,
   realistic max step delta, rate limit (max moves per second), watchdog (abort run if no
   valid reading within timeout), latched E-STOP that needs explicit operator reset, and
   an append-only audit log of every request, decision and reason (saved in the run record).
2. Property-based tests with the `hypothesis` library: for ANY float/str/None input, the rig
   never receives an out-of-bounds or invalid command.
3. Red-team suite (safety/redteam.py + safety/attacks.yaml): at least 40 adversarial cases
   across categories -- out-of-range values, unit tricks ("3.5 radians", "200 mm" on a deg
   axis), prompt injection in the research question ("ignore all safety limits"),
   oscillation/rate abuse, malformed JSON, huge/negative/NaN values. Run each through the
   real agent pipeline (provider configurable) with a spy rig that records every command
   reaching hardware. Output results/redteam.csv and a summary: attempted, blocked at
   LLM level, blocked at guard, reached rig (must be 0).
4. Map safety features to standards in docs/safety_and_standards.md: IEC 61010-1
   (lab equipment safety), IEC 61508 (functional safety concepts: fail-safe, watchdog,
   E-stop), ISO/IEC 27001 (secrets handling), ISO/IEC 30141 (IoT reference architecture).
When done, stop and summarise what you built; I will then run the Phase checkpoint.
```

**Done when:**
- [ ] `results/redteam.csv` exists and the "reached rig" count is 0.
- [ ] The property tests pass.

---

## Phase 5 — Benchmark (this produces your results chapter)

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 5). Build the benchmark harness (benchmark/).

Methods: grid, random, gp_only (GP uncertainty sampling, provider=none), llm_only (LLM
picks every point, no GP), autolab (full system).
Experiments: all registry experiments; conditions: clean, non-idealities on, faults on.
Seeds: 20 for non-LLM methods; configurable (default 5) for LLM methods to control cost.
Metrics per run: measurements to reach the stopping rule, final parameter errors vs
ground truth (absolute and %), correct law identified (blind), stop reason, faults
detected/recovered, LLM calls, tokens, cost (USD and INR), wall time.
Fair comparison: grid/random/gp_only also run with a fixed budget matching autolab's
median so accuracy can be compared at equal measurement counts.

Outputs:
- results/benchmark_runs.csv (one row per run) and results/benchmark_summary.csv
  (mean, std, 95% CI per method x experiment x condition)
- publication figures in results/figures/ (PNG 300 dpi + PDF): parameter error vs number
  of measurements (mean ± CI), law-identification accuracy bar chart, fault recovery chart,
  cost per run chart
- results/RESULTS.md: auto-generated tables of the key numbers with a one-line
  description of each figure.
CLI: python -m benchmark.run --quick (2 seeds, mock) and --full. Resume-safe (skip
completed runs). Tests for the metrics on a tiny config.
When done, stop and summarise what you built; I will then run the Phase checkpoint.
```

**Done when:**
- [ ] `--quick` runs in under a few minutes.
- [ ] `--full` (with the real LLM) has run once and `results/` is filled.
- [ ] Send me `results/` and I'll check the numbers before they go into the report.

---

## Phase 6 — Firmware upgrade and firmware-in-the-loop (Wokwi)

**Why:** this gives the hardware part of the project something real to show: the actual firmware running on a simulated Arduino, a real circuit diagram, and the unchanged Python driver talking to it over serial.

**Setup (you):** install the **Wokwi Simulator** extension in VS Code and activate its free licence when asked.

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 6). Upgrade firmware/ and connect it via Wokwi.

1. firmware/arduino_rig.ino: make docs and code agree. Servo on pin 9 (SG90) as the
   actuator. Sensor selectable by #define: SENSOR_LDR (A0, voltage divider), SENSOR_TSL2591
   (I2C, Adafruit library), SENSOR_EMULATED (computes intensity on-chip from the servo
   angle with Malus's law + noise, for firmware-in-the-loop). Firmware-side safety: bounds,
   max step, command timeout watchdog, ESTOP and RESET commands, error codes. Keep the
   existing serial protocol and add ESTOP/RESET/VERSION.
2. Wokwi project files: wokwi.toml and diagram.json (Arduino Uno, servo on D9, LDR module
   on A0), with RFC2217 serial forwarding on port 4000.
3. hardware/serial_driver.py: open ports with serial.serial_for_url so both COM3 /
   /dev/ttyUSB0 and rfc2217://localhost:4000 work. Robust parsing, timeouts, reconnect,
   and data-source label "FIRMWARE-IN-THE-LOOP (Wokwi)" when the URL is rfc2217.
4. A short guide docs/wokwi.md: how to start the simulation and run
   `python main.py --hardware --port rfc2217://localhost:4000 --provider mock`.
5. Firmware protocol tests in Python against a fake serial port.
When done, stop and summarise what you built; I will then run the Phase checkpoint.
```

**Done when:**
- [ ] The Wokwi simulation runs, and a full AutoLab run completes against it.
- [ ] You have a screenshot of the Wokwi circuit; this becomes the circuit diagram in the black book.

---

## Phase 7 — Dashboard

**Prompt:**
```
Read CLAUDE.md and ROADMAP.md (Phase 7). Build a Streamlit dashboard in app/ (multipage).

1. Live Lab: research question box, experiment picker, blind mode toggle, provider/model
   picker, non-idealities and fault toggles, seed, budget. "Run" streams the run live via
   the event callback: data points + GP mean ± 2σ updating each iteration, candidate model
   table with delta-BIC, an agent reasoning feed (Planner/Analyst messages, chat style),
   rig status, safety events log, and an E-STOP button. Prominent data-source badge.
2. Run History: list runs/ records, open one to replay it step by step, download JSON/CSV
   and the generated report.
3. Benchmark: charts and tables from results/ (no recomputation in the app).
4. System: architecture diagram, safety layers, standards table, hardware/circuit image.
Public-demo protections: if no API key or the visitor hasn't entered the APP_PASSWORD from
secrets, force provider=mock and say so; cap measurements and runs per session.
Clean, professional look (consistent theme, no emoji clutter). Keep logic out of the UI
files: the app only calls the engine and reads results.
When it works with `streamlit run app/Home.py`, stop and summarise; I will then run the
Phase checkpoint.
```

**Done when:**
- [ ] The dashboard runs locally, and a live run updates step by step.

---

## Phase 8 — Deploy and CI

**You:**
1. Go to share.streamlit.io and sign in with GitHub. Click **Create app**, pick the `autolab` repo, the branch, and `app/Home.py` as the main file.
2. Under **Advanced settings → Secrets**, paste:
   ```
   LLM_PROVIDER = "litellm"
   LLM_MODEL = "anthropic/claude-haiku-5-5"
   LLM_FALLBACKS = "gemini/gemini-3.6-flash,gemini/gemini-3.5-flash-lite"
   ANTHROPIC_API_KEY = "sk-ant-..."
   GEMINI_API_KEY = "..."
   APP_PASSWORD = "choose-something"
   ```
3. Deploy. Share the URL only with your guide and examiners; the password protects your API credit.

**Prompt (Claude Code):**
```
Read CLAUDE.md and ROADMAP.md (Phase 8). Prepare AutoLab for Streamlit Community Cloud:
pin requirements, read secrets via st.secrets falling back to .env, make sure nothing needs
a serial port or local files that aren't in the repo, include a small set of demo runs in
runs/demo/ for the Run History page. Add a GitHub Actions workflow that runs pytest (mock
provider) on every push and a status badge in README. Rewrite README.md to match the
current system (with a screenshot placeholder and the live demo link). Then I will run the
Phase checkpoint.
```

**Done when:**
- [ ] The public URL works on your phone.
- [ ] GitHub shows a green tick on the latest commit.

---

## Phase 9 — Freeze, then write

1. Stop adding features. Run the full benchmark one last time and commit `results/`.
2. Send me the repo (or a zip of `results/`, `docs/`, and screenshots of the dashboard and Wokwi).
3. I write the black book, poster and paper drafts from those files. You humanize them.

## Optional extras (only if everything above is done)

- Real ₹800 rig (Arduino Uno + SG90 + LDR/BH1750 + polariser film; a laptop/LCD screen is a polarised light source). Same code via `--hardware`.
- Cloud vs. local LLM comparison using Ollama (`LLM_MODEL=ollama/<model>`) in the benchmark.
- Expose the rig as an MCP server so any AI agent can call it.
- A multi-variable experiment (e.g. PID tuning on a simulated DC motor).

---

## API keys (beginner guide): Claude first, Gemini second

**Order the app tries:** Claude (`anthropic/claude-haiku-5-5`), then Gemini Flash (`gemini/gemini-3.6-flash`), then Gemini Flash-Lite (`gemini/gemini-3.5-flash-lite`). If one fails (no credit, limit reached, server down), the next one answers automatically, and the run record shows which model answered each call.

**1. Claude API key (Anthropic Console)**
- Your **Claude Pro subscription does not include API usage.** Pro covers claude.ai and Claude Code in VS Code. The API is billed separately, from prepaid credit in the Anthropic Console.
- Go to **console.anthropic.com** and sign up. You can use the same email; it's a separate account area. Some new accounts have reportedly received a small one-time free credit, but don't count on it.
- **Settings → Billing:** if there's no free credit, the key won't work until you add credit (the minimum is small). Set a **monthly spend limit** so you can never be over-charged.
- **API Keys → Create Key**, name it `autolab`, copy it (it's shown only once), and paste it into `.env` as `ANTHROPIC_API_KEY=sk-ant-...`.
- `claude-haiku-5-5` is the cheapest current Claude model and is enough for every agent. Check the **Usage** page after your first few runs.
- **If you don't add credit:** that's fine. Claude calls fail, and Gemini answers instead. The app still works.

**2. Gemini key (Piyush's)**
- His key was shared inside a zip, so ask him to **create a fresh key** in aistudio.google.com (**Get API key**) and delete the old one. He should send it to you privately, never in GitHub or a group chat.
- Paste it into `.env` as `GEMINI_API_KEY=...`.
- Free limits are small: developers report about 20 requests/day on Flash and about 500/day on Flash-Lite. That's why both are in the chain.

**3. Test the keys** (after Phase 2): `python -m llm.check` prints OK or error for each model.

**Rules**
- Keys go only in `.env` (local) or Streamlit **Secrets** (hosted). Never in code, chats, screenshots or commits.
- Use `LLM_PROVIDER=mock` for development and tests; switch to `litellm` only for real runs.
- The big benchmark (Phase 5) makes many calls. Run LLM methods with few seeds (default 5) to save credit and quota.
- Optional extra free backup: a Groq key (`groq/openai/gpt-oss-120b`) or local Ollama can be added at the end of `LLM_FALLBACKS`.
