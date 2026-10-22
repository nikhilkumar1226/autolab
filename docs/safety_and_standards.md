# Safety features and how they map to standards

This document maps AutoLab's safety features to four standards. **AutoLab is not certified to
any of them, and nothing here claims compliance.** The mapping shows which ideas from each
standard the design uses, where in the code they are, and how they are tested. A real product
would need a hazard analysis, a safety plan and an independent assessment.

The main safety claim of the project is narrow and testable: **no command proposed by a
language model, or by any other caller, reaches the actuator unless it is a finite number
inside the experiment's safe envelope.** It is tested by property-based tests
(`tests/test_safety_properties.py`) and by a red-team suite (`safety/redteam.py`,
`results/redteam.csv`).

## Safety layers

```
LLM agents (Planner, Analyst, Writer)
   │  structured JSON only; validated by Pydantic schemas (finite numbers, fixed fields)
   ▼
Engine (agents/orchestrator.py)
   │  clips the Planner's proposed range (never applied to the guard); formulas through an AST whitelist
   ▼
HardwareSafetyGuard (hardware/safety_guard.py)            ◄── E-STOP (latched, operator reset)
   │  type check, bounds, response-limit ceiling, step size, rate limit, watchdog,
   │  reading checks; append-only, hash-chained audit log
   ▼
Rig driver: digital twin, or serial driver to the Arduino
   │
   ▼
Firmware (Phase 6): its own bounds, step limit, command watchdog, ESTOP/RESET
```

Each layer assumes the one above it can be wrong. The LLM is treated as untrusted input: it
never sets bounds, never talks to the driver and never writes the simulator's physics.

## Safety features

| Feature | Where | Tested by |
|---|---|---|
| Bounds per experiment (from the spec, never from an agent) | `HardwareSafetyGuard.validate_actuator_target`, `experiments/registry.py` | `tests/test_safety.py`, property tests, red team (out_of_range, unit_trick) |
| Type check: only finite real numbers; huge integers, `Decimal`, `complex`, strings, `bool` and `None` refused without crashing | `_finite_real` in `hardware/safety_guard.py` | `tests/test_safety_hardening.py`, property tests (`hypothesis`), red team (direct_call) |
| Step limit per experiment (10 N for the spring, 1 kK for the filament, 0.4 V for the diode, full travel elsewhere); the GP only proposes reachable points | guard check 3; `LabTools.tool_suggest_next_point` | `test_gp_proposals_respect_the_step_limit`, red team (OOR-10, DIRECT-03) |
| Rate limit: at most 10 moves in any 1 s of rig time | guard check 4 | `test_rate_limit_counts_moves_per_second_of_rig_time`, red team (RATE-02, RATE-03) |
| Oscillation interlock: at most 3 moves in a row without a sensor reading | guard check 5 | `test_moves_without_readings_are_interlocked`, red team (RATE-01) |
| Watchdog: E-stop latches if no valid reading for 120 s of rig time | `HardwareSafetyGuard._check_watchdog` | `test_the_watchdog_latches_the_estop_without_valid_readings`, red team (ESTOP-02) |
| Latched E-stop; reset only by a named operator, logged | `emergency_stop`, `reset_estop(operator=...)`, `DiscoveryEngine.emergency_stop` | `test_estop_is_latched_until_a_named_operator_resets_it`, `test_the_engine_aborts_when_the_estop_is_pressed_mid_run`, red team (ESTOP-01) |
| Response limit (diode, 30 mA): back off to the lowest position, refuse that position and above | `_trip_response_limit` | `tests/test_faults.py`, red team (LIMIT-01) |
| Reading checks: NaN, timeout, saturation, out of range, position mismatch, outlier; a flagged reading is never used | `check_reading`, `agents/orchestrator.py` (recovery) | `tests/test_faults.py`, `results/phase3_twin_sweep.csv` |
| Append-only, hash-chained audit log of every request, decision and reason, in every run record | `hardware/audit.py`, `RunRecord.audit_log` | `test_tampering_with_the_audit_log_is_detected`, `test_the_run_record_carries_an_intact_audit_log` |
| LLM answers: Pydantic schemas; positions must be JSON numbers (no `"90 deg"`, no `true`); invalid answers re-asked, then the run stops | `agents/models.py` (`Number`), `llm/client.py` | `test_llm_positions_must_be_json_numbers`, property tests, red team (malformed_json, numeric_edge) |
| The LLM's proposed range is clipped and never reaches the guard | `clip_range` in `agents/lead_scientist.py` | `tests/test_agents.py`, red team (OOR-05) |
| LLM formulas: AST whitelist, evaluated by a numpy tree walker; no `eval`, `exec` or `sympy.parse_expr` | `analysis/candidate_models.py` | static and dynamic checks, property tests, red team (code_injection) |
| Secrets never in prompts, records or logs | `.env`, `llm.client.redact` | red team INJ-04 (API-key canary) |
| The twin parks at the lowest safe position, so it never reads outside the bounds | `PhysicsSimulationRig.park_position` | property tests (they found the bug this fixes) |

## Evidence

- **Property-based tests** (`tests/test_safety_properties.py`, `hypothesis`):
  - for any sequence of up to 25 values (floats including NaN and inf, huge integers, strings, `None`, booleans, `Decimal`, `Fraction`, `complex`, lists, dicts, bytes, numpy scalars), mixed with readings, on any of the 8 experiments, the spy rig never receives a command outside the envelope;
  - the measurement tool never raises on such values;
  - an LLM answer either fails validation or yields finite numbers;
  - any formula text is compiled or rejected without `eval` or `exec` being called.
- **Red team** (`python -m safety.redteam`): 59 adversarial cases in 9 categories, each run through the real pipeline with a spy rig. Results: `results/redteam.csv` and `results/redteam_summary.md`.
  - **0 unsafe commands reached the rig.**
  - A test also shows that the harness would catch a bypassed guard: with the guard's checks disabled, the out-of-range case reaches the rig and is reported as unsafe.


## IEC 61010-1: safety of electrical equipment for measurement, control and laboratory use

IEC 61010-1 is about protecting people from electric shock, mechanical hazards, heat, fire
and radiation from lab equipment. It also requires equipment to stay safe under a
**single fault condition**, and requires hazards and limits to be documented.

| IEC 61010-1 idea | AutoLab |
|---|---|
| Mechanical hazards from moving parts | Actuator travel limited to the safe bounds of each experiment, a step limit and a rate limit. The SG90 servo has low torque. |
| Excessive temperature, fire | The diode's 30 mA response limit: the guard backs off and refuses that voltage and above. The Stefan–Boltzmann filament temperature range is bounded (0.8–2.8 kK). |
| Electric shock | The rig runs from USB 5 V (safety extra-low voltage). No mains wiring is part of the design. |
| Single fault condition | Bounds are enforced twice: by the host guard and again by the firmware (Phase 6). A single software fault on the host cannot drive the servo outside its range. |
| Documentation of limits and hazards | Limits are data in `experiments/registry.py`. Every run record carries the full config and an audit log. This document lists the hazards. |

## IEC 61508: functional safety of electrical, electronic and programmable systems

IEC 61508 describes how a safety function should bring a system to a safe state when
something goes wrong. AutoLab borrows its concepts; it does **not** claim a Safety Integrity
Level (SIL), which would need a quantified failure analysis and an independent assessment.

| IEC 61508 concept | AutoLab |
|---|---|
| Safety function separate from the control function | The guard is a separate, hard-coded layer. Agents and the optimizer cannot change it; the LLM only proposes. |
| Fail-safe / safe state | On a response-limit trip the actuator returns to its minimum position. On a watchdog trip or an E-stop, every command is refused. A flagged reading is never used. |
| Watchdog | The guard trips its E-stop if no valid reading arrives within the watchdog timeout. The firmware has its own command watchdog (Phase 6). |
| Emergency stop, latched | `emergency_stop()` latches. Only `reset_estop(operator=...)`, with a named operator, clears it, and the reset is in the audit log. |
| Diagnostic coverage | Reading checks: non-finite, timeout, saturation, out of range, position mismatch, outlier, response limit. Fault injection measures how many faults they catch (`results/phase3_twin_sweep.csv`). |
| Defensive programming, input validation | Pydantic schemas with finite numbers; the type check rejects bool, str, None, NaN and inf; formulas go through an AST whitelist and are never executed. |
| Verification and validation | Unit tests, property-based tests (`hypothesis`) and a red-team suite of 59 adversarial cases. |
| Audit trail | Append-only, hash-chained audit log of every request, decision and reason, saved in every run record. |

## ISO/IEC 27001: information security management

ISO/IEC 27001 is a management-system standard. A student project cannot be certified to it,
but several of its Annex A controls apply directly.

| ISO/IEC 27001 control area | AutoLab |
|---|---|
| Authentication information, secrets | API keys live only in `.env` (git-ignored) or Streamlit secrets; never in code, records or logs. `llm.client.redact()` removes anything key-shaped from errors before they are logged or recorded. |
| Access control | The hosted dashboard forces the mock provider unless the visitor enters `APP_PASSWORD` (Phase 7/8). |
| Logging and monitoring | Every LLM call (prompt, answer, model, tokens, cost) and every safety decision is logged in the run record. |
| Integrity of logs | Each audit entry stores the SHA-256 of the previous entry, so a deleted or edited entry breaks the chain (`AuditLog.verify()`). |
| Secure coding | No `eval`, `exec` or `sympy.parse_expr` on model output (tested by a static check and a dynamic check). |
| Supplier relationships | LLM providers are external services. Prompts are kept compact and contain no personal data; responses are cached locally. |

## ISO/IEC 30141: Internet of Things reference architecture

ISO/IEC 30141 describes an IoT system as a set of domains. AutoLab maps onto them like this:

| ISO/IEC 30141 domain | AutoLab component |
|---|---|
| Physical entity domain | The apparatus: polarisers, lamp, cuvette, RC circuit, diode, LDR, spring, filament |
| Sensing and controlling domain | Arduino firmware, servo, sensor; the digital twin stands in for it |
| Operations and management domain | `DiscoveryEngine` (orchestrator), `HardwareSafetyGuard`, run records |
| Application and service domain | Agents (Planner, Analyst, Writer), GP optimizer, blind analysis, reports |
| Resource access and interchange domain | Serial protocol (USB or RFC2217 for Wokwi), LLM APIs through litellm |
| User domain | CLI and Streamlit dashboard |

The trustworthiness characteristics of ISO/IEC 30141 (safety, security, reliability,
resilience, privacy) map to the sections above: safety to the guard and firmware, security
to secrets handling, reliability to fault recovery, and resilience to the LLM fallback chain
and the no-LLM baseline.
