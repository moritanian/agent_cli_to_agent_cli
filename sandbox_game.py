"""Multi-agent sandbox simulation driven by CLI-backed LLM agents.

Each agent decides its action (move or talk) by querying a Gemini CLI model.
The simulation enforces grid boundaries, prevents collisions, and delivers
messages between adjacent agents. State snapshots are printed every turn.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from cli_clients import GeminiCliChatCompletionClient


ActionDict = Dict[str, str]


@dataclass
class AgentState:
    name: str
    controller: AssistantAgent
    x: int
    y: int
    inbox: Optional[str] = None
    last_action: Optional[ActionDict] = field(default=None, repr=False)

    @property
    def position(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}


def _build_system_prompt(persona: str, partner_name: str) -> str:
    return (
        f"{persona} 会話のパートナーは{partner_name}です。"
        "与えられた話題や状況について親しい仲間として語り合い、自分の考えや体験を一人称で共有してください。"
        f"{partner_name}の名前を含めて直接語りかける1段落以内の日本語で返答し、AIである旨や第三者目線の解説、ユーザー向けのまとめは書かないでください。"
        "箇条書きやツール実行は避け、相手の発言に共感や質問、提案を添えて会話を前進させてください。"
        "観測データに含まれるlegal_actionsから必ず1つを選び、その内容と整合するJSONだけを返答してください。"
        "必ず以下のJSON形式のみで応答してください："
        '{"action": "move"|"talk"|"wait", "direction": "up|down|left|right", "target": "agent_name", "message": "text"} '
        "moveの場合はdirectionを設定し、talkの場合はtargetとmessageを設定してください。waitを選ぶ場合は他のフィールドを省略できます。"
    )


def _build_agent(persona: str, name: str, partner: str, debug: bool) -> AssistantAgent:
    return AssistantAgent(
        name=name,
        system_message=_build_system_prompt(persona, partner),
        model_client=GeminiCliChatCompletionClient(debug=debug),
    )


def _initial_positions(grid_size: int, agent_names: List[str]) -> Dict[str, Dict[str, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    random.shuffle(cells)
    return {name: {"x": pos[0], "y": pos[1]} for name, pos in zip(agent_names, cells)}


async def _query_action(agent: AgentState, observation: Dict[str, object]) -> str:
    prompt = (
        "現在の状況と選択可能な合法手をJSONで渡します。"
        " legal_actionsの中から必ず1つを選び、指示されたJSON形式のみで応答してください。\n"
        f"{json.dumps(observation, ensure_ascii=False)}"
    )
    result = await agent.controller.run(task=[TextMessage(content=prompt, source="user")])
    for message in reversed(result.messages):
        if hasattr(message, "to_text"):
            text = message.to_text()
        else:
            text = getattr(message, "content", "")
        if text:
            return text.strip()
    raise RuntimeError(f"{agent.name} did not return a textual action.")


def _extract_json_block(raw: str) -> Optional[str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    return match.group(0)


def _parse_action(raw: str) -> ActionDict:
    block = _extract_json_block(raw)
    if not block:
        return {"action": "wait"}
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return {"action": "wait"}
    action = data.get("action")
    if action not in {"move", "talk", "wait"}:
        return {"action": "wait"}
    if action == "move":
        direction = data.get("direction")
        if direction not in {"up", "down", "left", "right"}:
            return {"action": "wait"}
        return {"action": "move", "direction": direction}
    if action == "talk":
        target = data.get("target")
        message = data.get("message")
        if not isinstance(target, str) or not isinstance(message, str):
            return {"action": "wait"}
        return {"action": "talk", "target": target, "message": message}
    return {"action": "wait"}


def _move_delta(direction: str) -> tuple[int, int]:
    return {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }[direction]


def _is_adjacent(a: AgentState, b: AgentState) -> bool:
    return abs(a.x - b.x) + abs(a.y - b.y) == 1


def _legal_actions(agent: AgentState, agents: List[AgentState], grid_size: int) -> List[ActionDict]:
    legal: List[ActionDict] = [{"action": "wait"}]
    for direction, (dx, dy) in {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }.items():
        new_x = agent.x + dx
        new_y = agent.y + dy
        if (
            0 <= new_x < grid_size
            and 0 <= new_y < grid_size
            and not any(
                other.name != agent.name and other.x == new_x and other.y == new_y
                for other in agents
            )
        ):
            legal.append({"action": "move", "direction": direction})

    for other in agents:
        if other.name != agent.name and _is_adjacent(agent, other):
            legal.append({"action": "talk", "target": other.name})
    return legal


async def run_simulation(
    *,
    num_agents: int = 2,
    grid_size: int = 3,
    turns: int = 3,
    debug: bool = False,
) -> None:
    if num_agents != 2:
        raise ValueError("This prototype currently supports exactly 2 agents.")

    agent_names = [f"agent{i+1}" for i in range(num_agents)]
    positions = _initial_positions(grid_size, agent_names)

    personas = {
        "agent1": "あなたはエンジニアのAlexで、札幌出身。土地の食文化や海辺の街が好きで、休日には市場を散策するのが楽しみです",
        "agent2": "あなたはアウトドア派のBlairで、京都出身。山歩きや温泉巡り、カメラでの撮影が好きです",
    }

    controllers = {
        "agent1": _build_agent(personas["agent1"], "agent1", "agent2", debug),
        "agent2": _build_agent(personas["agent2"], "agent2", "agent1", debug),
    }

    agents = [
        AgentState(
            name=name,
            controller=controllers[name],
            x=positions[name]["x"],
            y=positions[name]["y"],
        )
        for name in agent_names
    ]

    for turn in range(1, turns + 1):
        print(f"\n=== Turn {turn} ===")
        print("Positions:", {agent.name: agent.position for agent in agents})

        for agent in agents:
            legal_actions = _legal_actions(agent, agents, grid_size)
            observation = {
                "you": agent.name,
                "positions": {state.name: state.position for state in agents},
                "grid_size": grid_size,
                "turn": turn,
                "legal_actions": legal_actions,
            }
            if agent.inbox:
                observation["message"] = agent.inbox
            agent.inbox = None

            print(f"\n[{agent.name}] thinking...")
            print(f"[{agent.name}] legal actions: {legal_actions}")
            raw_response = await _query_action(agent, observation)
            action = _parse_action(raw_response)
            if action["action"] == "move":
                allowed_moves = {
                    item["direction"]
                    for item in legal_actions
                    if item.get("action") == "move"
                }
                if action.get("direction") not in allowed_moves:
                    print(f"[{agent.name}] move direction not legal; waiting instead.")
                    action = {"action": "wait"}
            elif action["action"] == "talk":
                allowed_targets = {
                    item["target"]
                    for item in legal_actions
                    if item.get("action") == "talk"
                }
                if action.get("target") not in allowed_targets:
                    print(f"[{agent.name}] talk target not legal; waiting instead.")
                    action = {"action": "wait"}
            agent.last_action = action
            print(f"[{agent.name}] response: {raw_response}")
            print(f"[{agent.name}] parsed action: {action}")

            if action["action"] == "move":
                dx, dy = _move_delta(action["direction"])
                new_x = agent.x + dx
                new_y = agent.y + dy
                blocked = (
                    new_x < 0
                    or new_x >= grid_size
                    or new_y < 0
                    or new_y >= grid_size
                    or any(
                        other.name != agent.name and other.x == new_x and other.y == new_y
                        for other in agents
                    )
                )
                if blocked:
                    print(f"[{agent.name}] move blocked; staying at {agent.position}")
                else:
                    agent.x = new_x
                    agent.y = new_y
                    print(f"[{agent.name}] moved to {agent.position}")
            elif action["action"] == "talk":
                target_name = action["target"]
                target_agent = next((a for a in agents if a.name == target_name), None)
                if target_agent and _is_adjacent(agent, target_agent):
                    target_agent.inbox = action["message"]
                    print(
                        f"[{agent.name}] talked to {target_name}: {action['message']}"
                    )
                else:
                    print(
                        f"[{agent.name}] attempted to talk to {target_name}, but no adjacent agent was found."
                    )
            else:
                print(f"[{agent.name}] waits.")

        print("End of turn positions:", {agent.name: agent.position for agent in agents})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sandbox LLM agent simulation.")
    parser.add_argument("--grid", type=int, default=3, help="Grid size (grid x grid).")
    parser.add_argument("--agents", type=int, default=2, help="Number of agents.")
    parser.add_argument("--turns", type=int, default=3, help="Number of turns to simulate.")
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
        )
    )


if __name__ == "__main__":
    main()
