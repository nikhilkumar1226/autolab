# Handoff notes (overnight autonomous run, 2026-10-22)

Written for Nikhil when you wake up. Updated after every phase. The detailed "what changed / why / how verified" notes for each phase are in `docs/CHANGELOG.md`; this file is the short version plus everything that needs you.

Rules I followed: provider `mock` (or `none`) only, no real LLM or paid API was ever called, no API key was read or printed, `.env` was never staged.

## Status

| Phase | Branch | Tag | Tests | Done-when |
|---|---|---|---|---|
| 2 (fix) | `phase-2-llm-agents` | `phase-2.1` | 230 passed | 2 of 4 pass; the other 2 need your keys |

## Decisions I made

- **Phase 2 tag.** `phase-2` already existed on GitHub (your save point at `55b8dc1`). I did not move it, because moving a pushed tag rewrites what others have fetched. The fixed Phase 2 is tagged **`phase-2.1`**.
- **Branches.** I kept your pattern of one branch per phase. Each new phase branch starts from the previous one, so the last branch contains everything. `main` is still at the initial commit; I did not merge or open pull requests (that is your review step).
- **Formula whitelist (Phase 2 fix).** I implemented exactly the whitelist you gave. Two small choices:
  - Unary plus (`+x`) is rejected, because the list names unary minus only.
  - Python builtin names (`open`, `eval`, `id`, …) cannot be parameter names. The evaluator never looks names up in Python, so this is defence in depth and readability, not a hole being closed.
  - sympy used to simplify the expression to detect "does not depend on x", "parameter has no effect" and `1/0`. Without sympy these checks are numerical, at fixed probe points. Parameter kinds (scale / location / offset) come from numerical derivatives. See CHANGELOG "Phase 2 fix" for the one known limitation.
- **sympy removed.** No code uses it now, so I removed it from `requirements.txt` and uninstalled it from `.venv` (`pip install sympy` puts it back if you ever need it).

## What you must do yourself

1. **`.env` layout.** Your `.env` still has the old Groq/Ollama layout. Copy the layout from `.env.example` (Claude first, then the two Gemini models) and add `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. I did not touch `.env`.
2. **Phase 2 open Done-when items** (need keys):
   - `python -m llm.check` → should print OK for Claude and Gemini.
   - `python main.py --provider litellm` → check the run JSON in `runs/` shows the real prompts, responses, tokens and cost.

## Open questions

- None yet.
