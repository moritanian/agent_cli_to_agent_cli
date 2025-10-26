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
   shared conversation log.

## Customisation tips

- Set `GeminiCliChatCompletionClient(model=...)` if you want a specific Gemini model; leaving it `None` lets the CLI pick its default (usually Gemini 2.5 Pro).
- Append CLI switches via `extra_flags` (temperature, safety settings, etc.).
- Swap the hard-coded prompts in either script for your own workflow or plug
  the client into a richer `Team` configuration.
- `cli_clients.py` also provides `CLIChatCompletionClient` for building your own
  adapters and a `CodexCliChatCompletionClient` example; swap the factory in the
  scripts if you want to target a different CLI by default.
