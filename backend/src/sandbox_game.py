"""CLI runner for the sandbox simulation."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict

from sandbox_simulation import SandboxSimulation


async def run_simulation(
    *,
    num_agents: int = 2,
    grid_size: int = 3,
    turns: int = 3,
    debug: bool = False,
    seed: int | None = None,
    backend: str = "gemini",
    player_agent: bool = False,
) -> None:
    sim = SandboxSimulation(
        num_agents=num_agents,
        grid_size=grid_size,
        debug=debug,
        seed=seed,
        backend=backend,
        player_agent=player_agent,
    )
    snapshot = sim.reset()
    print("=== Initial State ===")
    print(_format_snapshot(snapshot))

    for _ in range(turns):
        turn_result = await sim.step()
        print(f"\n=== Turn {turn_result['turn']} ===")
        print(_format_turn(turn_result))


def _format_snapshot(snapshot: Dict[str, Any]) -> str:
    agents = ", ".join(
        f"{entry['name']}@({entry['position']['x']},{entry['position']['y']})"
        for entry in snapshot["agents"]
    )
    return f"Turn {snapshot['turn']} | Agents: {agents}"


def _format_turn(turn_result: Dict[str, Any]) -> str:
    lines: list[str] = []
    snapshot = turn_result["snapshot"]
    lines.append(_format_snapshot(snapshot))
    if turn_result["turnMessages"]:
        for msg in turn_result["turnMessages"]:
            lines.append(f"{msg['from']} -> {msg['to']}: {msg['message']}")
    for debug in turn_result["debug"]:
        lines.append(f"{debug['agent']} action: {debug['action']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sandbox LLM agent simulation.")
    parser.add_argument("--grid", type=int, default=3, help="Grid size (grid x grid).")
    parser.add_argument("--agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--turns", type=int, default=3, help="Number of turns to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for initial placement.")
    parser.add_argument(
        "--backend",
        default="gemini",
        choices=["gemini", "codex", "mock"],
        help="CLI backend to use for LLM calls.",
    )
    parser.add_argument(
        "--player",
        action="store_true",
        help="Include a player-controlled adventurer (last agent).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable CLI debug logging for agent model clients.",
    )
    args = parser.parse_args()

    asyncio.run(
        run_simulation(
            num_agents=args.agents,
            grid_size=args.grid,
            turns=args.turns,
            debug=args.debug,
            seed=args.seed,
            backend=args.backend,
            player_agent=args.player,
        )
    )


if __name__ == "__main__":
    main()
