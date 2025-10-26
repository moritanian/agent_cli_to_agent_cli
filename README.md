# AutoGen × Gemini CLI Sample

A minimal AutoGen AgentChat (v0.10) proof-of-concept that shells out to the
Gemini CLI in non-interactive mode (`gemini -p`) via a custom
`ChatCompletionClient`. Two entry points are provided:

- `main_single_agent.py` — original single-assistant demo that answers a prompt.
- `main.py` — new round-robin sample where two Gemini-backed agents collaborate.

## Prerequisites

- Python 3.10+ with `pyautogen>=0.10.0` installed (`python3 -m pip install --user pyautogen`).
- Gemini CLI available on `PATH` (check with `gemini --help`).
- Gemini API credentials exported so the CLI can authenticate, e.g. `export GEMINI_API_KEY=...`.
- Outbound network access permitted for the CLI invocation.

## Usage

```bash
# Single-assistant ping
python3 main_single_agent.py

# Two-agent collaboration
python3 main.py
```

Both scripts share the same Gemini CLI-backed client defined in `cli_clients.py`.
The multi-agent version runs two `AssistantAgent`s in sequence (planner →
executor), passing the first agent's reply as context for the second so you can
observe a lightweight two-party exchange.

## Sandbox Web UI

This repository also includes an interactive sandbox viewer for debugging
multi-agent runs:

1. Start the FastAPI backend:
   ```bash
   cd backend
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

- From the CLI, run `python sandbox_game.py --backend codex|gemini|mock` to
  switch LLM providers; the web UI exposes the same toggle in the control
  panel (the mock backend picks random actions for quick smoke tests).
- Both `GeminiCliChatCompletionClient` and `CodexCliChatCompletionClient` accept
  extra CLI flags if you need to tune temperature, model IDs, or safety settings.
- Swap the hard-coded prompts in `main.py` or wire the shared `SandboxSimulation`
  into your own AutoGen team configuration to experiment with different agent
  personas.
