# Sandbox Agent Playground

The primary deliverable in this repository is an interactive sandbox web UI for
observing and debugging multi-agent conversations that run on top of AutoGen
AgentChat (v0.10). The React frontend renders a glowing battle grid, per-turn
conversation log, and detailed debug feed while the FastAPI backend coordinates
LLM-powered agents (Gemini, Codex, or a deterministic mock). Helpers and CLI
demos remain available, but the browser-based playground is considered the core
experience.

## Architecture Snapshot

- **Frontend (`frontend/src`)** – Vite + React dashboard that drives the sandbox UI. It advances the simulation turn-by-turn via HTTP calls, shows board state, and surfaces prompts/responses with live styling.
- **Backend (`backend/src`)** – FastAPI service (`main.py`) wrapping `SandboxSimulation`, which manages agent state, legal actions, player turns, and backend selection. LLM clients and CLI demos live alongside the service.
- **CLI tooling** – Modules such as `two_agent_demo.py`, `single_agent_demo.py`, and `sandbox_game.py` reuse the same simulation logic for terminal-based experiments.

For contributor workflow expectations (coding style, test commands, and release etiquette) see `AGENTS.md`.

## Sandbox Web UI Specification

- **Core loop** – Frontend triggers `/reset`, then `/step` for each turn. Backend returns snapshots with agent positions, conversation log, debug prompts, and whether a player-controlled action is required.
- **Board view** – Renders the sandbox grid with responsive sizing (supports ≥4×4) and themed agent tiles. Each occupied cell displays icon, title, and the agent’s latest spoken line for the current turn.
- **Control panel** – Allows configuration of grid size (2–8), agent count (2–6), debug flag, backend choice (`gemini`, `codex`, `mock`), random seed, and toggling a player-controlled agent. Reset reapplies settings; Step advances the simulation.
- **Conversation log** – Lists interactions chronologically (Turn N, speaker → target, message). Automatically scrolls when long.
- **Debug panel** – Accordion of per-turn prompt/response payloads, legal action lists, parsed actions, and optional notes. Scrollable to prevent layout overflow.
- **Player actions** – When the simulation asks for player input, the UI shows selectable moves and talk options (with editable message drafts) for the designated agent.
- **Backends** – Gemini and Codex modes shell out to their respective CLIs; Mock mode returns deterministic random actions for quick smoke tests. Backend choice propagates to both logs and board badges.
  
Implementation details may evolve; this section captures the intended behaviour at a high level.

## Prerequisites

- [`uv` 0.4+](https://docs.astral.sh/uv/) (installs and manages isolated Python environments).
- Python 3.10+ (automatically provisioned by `uv` if missing locally).
- Gemini CLI available on `PATH` (check with `gemini --help`).
- Gemini API credentials exported so the CLI can authenticate, e.g. `export GEMINI_API_KEY=...`.
- Outbound network access permitted for the CLI invocation.

## Usage

```bash
cd backend

# Install/update the virtual environment defined in pyproject.toml
uv sync

# Single-assistant ping
uv run python -m single_agent_demo

# Two-agent collaboration
uv run python -m two_agent_demo
```

Both scripts share the same Gemini CLI-backed client defined in `cli_clients.py`.
The multi-agent version runs two `AssistantAgent`s in sequence (planner →
executor), passing the first agent's reply as context for the second so you can
observe a lightweight two-party exchange.

## Running the Sandbox UI

This repository also includes an interactive sandbox viewer for debugging
multi-agent runs:

1. Start the FastAPI backend:
   ```bash
   cd backend
   uv sync
   uv run serve --reload
   ```
2. Install and launch the Vite + React frontend:
   ```bash
   cd frontend
   yarn install
   yarn dev
   ```
3. Open the printed local URL (default `http://localhost:5173`) to step the
   simulation turn-by-turn, inspect agent prompts/responses, and review the
   shared conversation log. Use the controls at the top to adjust grid size,
   agent count, or switch between the Gemini, Codex, and in-memory Mock LLM
   backends before hitting Reset.

## Customisation tips

- From the CLI, run `uv run python -m sandbox_game --backend codex|gemini|mock` to
  switch LLM providers; the web UI exposes the same toggle in the control
  panel (the mock backend picks random actions for quick smoke tests).
- Enable "Add player-controlled agent" in the web UI (or pass `--player` to the
  CLI) to manually choose actions from the legal move list each turn.
- Both `GeminiCliChatCompletionClient` and `CodexCliChatCompletionClient` accept
  extra CLI flags if you need to tune temperature, model IDs, or safety settings.
- Swap the hard-coded prompts in `two_agent_demo.py` or wire the shared `SandboxSimulation`
  into your own AutoGen team configuration to experiment with different agent
  personas.
