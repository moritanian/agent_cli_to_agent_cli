# Repository Guidelines

This note complements the high-level context in `README.md`. For the sandbox UI specification and environment setup, start there; use this guide for contributor expectations and daily workflows.

## Project Structure & Module Organization

- `frontend/` contains the Vite + React sandbox UI and is the primary application for this repository. Components, hooks, and styles reside under `frontend/src` (see README for the feature overview).
- `backend/` houses the FastAPI service plus reusable LLM clients and CLI demos (all code lives under `backend/src`). `sandbox_simulation.py` is the authoritative engine for multi-agent state and turn resolution.
- Top-level files (`README.md`, `AGENTS.md`, `TODO.md`) document process and outstanding work; they should remain light on implementation detail.

## Build, Test, and Development Commands

Run everything from inside `backend/` or `frontend/` with `uv` and `yarn` respectively:
- Backend environment sync: `uv sync`
- FastAPI server: `uv run serve --reload`
- CLI demos: `uv run python -m two_agent_demo` (or `single_agent_demo`, `sandbox_game`, etc.)
- Backend tests: `uv run pytest`
- Frontend dev server: `yarn install` (first run) then `yarn dev`
- Frontend build check: `yarn build`

## Coding Style & Naming Conventions

Python code follows 4-space indentation, `snake_case` names, and type hints. Shared clients inherit from `CLIChatCompletionClient`; keep constructor keyword arguments (`make_argv`, `parse_response`, `extra_flags`) consistent to simplify swapping backends. Use concise docstrings at the module/class level and reserve inline comments for logic that is not self-evident.

React components live in `frontend/src`, use functional components with hooks, and keep JSX presentational logic close to domain utilities. Co-locate styles in `App.css` unless a component genuinely warrants modular CSS.

## Testing Guidelines

Unit tests run via `uv run pytest` and currently live in `backend/tests`. Prefer mocking LLM calls with `MockCliChatCompletionClient` when writing new tests. Frontend changes rely on manual verification—call out expected visual checks (board layout, conversation log scroll, debug panel behaviour) in your PR description until automated UI coverage is introduced.

## Commit & Pull Request Guidelines

Keep commits focused and descriptive (imperative tense, <75 characters). Mention the subsystem in the subject when helpful (e.g., `backend`, `frontend`). Provide a short body describing motivation + validation steps if not obvious. Do not push new commits without explicit approval from the requestor when they are actively guiding the work.
