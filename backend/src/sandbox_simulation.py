"""Core sandbox simulation logic for multi-agent LLM coordination."""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

# Make project root (containing cli_clients) available when installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_clients import GeminiCliChatCompletionClient

ActionDict = Dict[str, str]


@dataclass
class AgentState:
    name: str
    controller: AssistantAgent
    x: int
    y: int
    inbox: Optional[Dict[str, str]] = None
    last_action: Optional[ActionDict] = field(default=None, repr=False)

    @property
    def position(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}


def _build_system_prompt(persona: str, roster: str) -> str:
    return (
        f"{persona} Your teammates are {roster}. "
        "Speak like a friendly adventurer, sharing your thoughts in the first person. "
        "When you choose a talk action, pick one of the characters listed in legal_actions and greet them by name in a short English paragraph. "
        "Do not mention that you are an AI, write third-person commentary, or summarise for the user. "
        "Avoid bullet points and tool usage; respond with empathy, questions, or suggestions that move the party forward. "
        "You must select exactly one option from legal_actions and return JSON that matches it. "
        'Return JSON only in the form {"action": ..., "direction"|"target"|"message": ...}. '
        "For move, set direction. For talk, set target and message. For wait, omit the other fields."
    )


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


def _move_delta(direction: str) -> Tuple[int, int]:
    return {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }[direction]


def _is_adjacent(a: AgentState, b: AgentState) -> bool:
    return abs(a.x - b.x) + abs(a.y - b.y) == 1


def _legal_actions(agent: AgentState, agents: Sequence[AgentState], grid_size: int) -> List[ActionDict]:
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


class SandboxSimulation:
    """Manages LLM-backed agents inside a grid sandbox."""

    def __init__(
        self,
        *,
        num_agents: int = 2,
        grid_size: int = 3,
        debug: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if num_agents < 2:
            raise ValueError("This prototype currently supports at least 2 agents.")
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.debug = debug
        self.seed = seed
        self._rng = random.Random(seed)

        self.turn = 0
        self.agents: List[AgentState] = []
        self.conversation_log: List[Dict[str, str]] = []
        self.debug_history: List[Dict[str, object]] = []
        self.agent_profiles: Dict[str, Dict[str, str]] = {}
        self._personas_pool: List[Dict[str, str]] = [
            {
                "title": "Alex",
                "icon": "🛡️",
                "color": "#8ecae6",
                "glow": "rgba(142, 202, 230, 0.6)",
                "persona": "You are Alex, an engineer from Sapporo who loves seaside towns and bustling markets.",
            },
            {
                "title": "Blair",
                "icon": "🗡️",
                "color": "#f9a03f",
                "glow": "rgba(249, 160, 63, 0.6)",
                "persona": "You are Blair, an adventurer from Kyoto who enjoys mountain hikes, hot springs, and photography.",
            },
            {
                "title": "Kai",
                "icon": "🪄",
                "color": "#bb6bd9",
                "glow": "rgba(187, 107, 217, 0.6)",
                "persona": "You are Kai, a travelling arcane researcher who studies starlit skies and ancient manuscripts.",
            },
            {
                "title": "Mira",
                "icon": "🏹",
                "color": "#6ee7b7",
                "glow": "rgba(110, 231, 183, 0.55)",
                "persona": "You are Mira, a ranger honed by the forest with keen insight and swift judgement.",
            },
            {
                "title": "Ren",
                "icon": "⚒️",
                "color": "#f97316",
                "glow": "rgba(249, 115, 22, 0.5)",
                "persona": "You are Ren, a tinkerer who cannot resist dismantling mysterious devices to learn their secrets.",
            },
        ]

    def reset(self) -> Dict[str, object]:
        """Initialise positions and re-create agents."""
        self.turn = 0
        self.conversation_log = []
        self.debug_history = []
        self.agent_profiles = {}

        positions = self._initial_positions()
        controllers = self._build_controllers()
        self.agents = [
            AgentState(
                name=name,
                controller=controllers[name],
                x=positions[name]["x"],
                y=positions[name]["y"],
            )
            for name in sorted(controllers.keys())
        ]
        return self.snapshot()

    def _initial_positions(self) -> Dict[str, Dict[str, int]]:
        cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self._rng.shuffle(cells)
        agent_names = [f"agent{i+1}" for i in range(self.num_agents)]
        return {name: {"x": pos[0], "y": pos[1]} for name, pos in zip(agent_names, cells)}

    def _build_controllers(self) -> Dict[str, AssistantAgent]:
        controllers: Dict[str, AssistantAgent] = {}
        agent_names = [f"agent{i+1}" for i in range(self.num_agents)]
        for index, name in enumerate(agent_names):
            profile = self._personas_pool[index % len(self._personas_pool)].copy()
            profile.setdefault("title", f"Agent {index + 1}")
            profile.setdefault("icon", "★")
            profile.setdefault("color", "#7dd3fc")
            profile.setdefault("glow", "rgba(125, 211, 252, 0.55)")
            profile.setdefault("persona", f"Speak naturally as {profile['title']} with your party.")
            self.agent_profiles[name] = profile

        for name in agent_names:
            roster = [
                f"{self.agent_profiles[other]['title']} ({other})"
                for other in agent_names
                if other != name
            ]
            roster_desc = ", ".join(roster)
            persona = self.agent_profiles[name]["persona"]
            controllers[name] = AssistantAgent(
                name=name,
                system_message=_build_system_prompt(persona, roster_desc),
                model_client=GeminiCliChatCompletionClient(debug=self.debug),
            )
        return controllers

    def snapshot(self) -> Dict[str, object]:
        return {
            "turn": self.turn,
            "gridSize": self.grid_size,
            "agents": [
                {"name": agent.name, "position": agent.position}
                for agent in self.agents
            ],
            "traits": self.agent_profiles,
            "messages": list(self.conversation_log),
        }

    async def step(self) -> Dict[str, object]:
        if not self.agents:
            raise RuntimeError("Simulation not initialised. Call reset() first.")

        self.turn += 1
        turn_messages: List[Dict[str, str]] = []
        turn_debug: List[Dict[str, object]] = []

        for agent in self.agents:
            legal_actions = _legal_actions(agent, self.agents, self.grid_size)
            for entry in legal_actions:
                if entry["action"] == "talk":
                    profile = self.agent_profiles.get(entry["target"], {})
                    entry["target_title"] = profile.get("title", entry["target"])
            observation: Dict[str, object] = {
                "you": agent.name,
                "positions": {state.name: state.position for state in self.agents},
                "grid_size": self.grid_size,
                "turn": self.turn,
                "legal_actions": legal_actions,
                "traits": self.agent_profiles,
            }
            if agent.inbox:
                observation["message"] = agent.inbox
            agent.inbox = None

            prompt = self._build_prompt(observation)
            raw_response = await self._query_action(agent, prompt)
            action = _parse_action(raw_response)

            action = self._enforce_legality(action, legal_actions, agent.name)
            agent.last_action = action

            debug_entry = {
                "agent": agent.name,
                "prompt": prompt,
                "response": raw_response,
                "legal_actions": legal_actions,
                "action": action,
            }
            turn_debug.append(debug_entry)

            if action["action"] == "move":
                dx, dy = _move_delta(action["direction"])
                agent.x += dx
                agent.y += dy
            elif action["action"] == "talk":
                target_name = action["target"]
                target_agent = next((a for a in self.agents if a.name == target_name), None)
                if target_agent and _is_adjacent(agent, target_agent):
                    payload = {
                        "from": agent.name,
                        "to": target_name,
                        "message": action["message"],
                        "turn": self.turn,
                    }
                    target_agent.inbox = {"from": agent.name, "message": action["message"]}
                    turn_messages.append(payload)
                    self.conversation_log.append(payload)
                else:
                    debug_entry["notes"] = "Talk target invalid or not adjacent."
                    agent.last_action = {
                        "action": "wait",
                        "notes": "Talk target invalid or not adjacent.",
                    }
                    debug_entry["action"] = agent.last_action
            else:
                pass

        snapshot = self.snapshot()
        turn_result = {
            "turn": self.turn,
            "snapshot": snapshot,
            "turnMessages": turn_messages,
            "debug": turn_debug,
        }
        self.debug_history.append(turn_result)
        return turn_result

    def _build_prompt(self, observation: Dict[str, object]) -> str:
        return (
            "You will receive the current situation and the available legal actions as JSON. "
            "Choose exactly one entry from legal_actions and respond only with the specified JSON shape.\n"
            f"{json.dumps(observation, ensure_ascii=False)}"
        )

    async def _query_action(self, agent: AgentState, prompt: str) -> str:
        result = await agent.controller.run(
            task=[TextMessage(content=prompt, source="user")]
        )
        for message in reversed(result.messages):
            if hasattr(message, "to_text"):
                text = message.to_text()
            else:
                text = getattr(message, "content", "")
            if text:
                return text.strip()
        raise RuntimeError(f"{agent.name} did not return a textual action.")

    def _enforce_legality(
        self,
        action: ActionDict,
        legal_actions: List[ActionDict],
        agent_name: str,
    ) -> ActionDict:
        if action["action"] == "move":
            allowed = {
                entry["direction"]
                for entry in legal_actions
                if entry.get("action") == "move"
            }
            if action.get("direction") not in allowed:
                return {"action": "wait", "notes": f"Illegal move rejected for {agent_name}"}
        elif action["action"] == "talk":
            allowed = {
                entry["target"]
                for entry in legal_actions
                if entry.get("action") == "talk"
            }
            if action.get("target") not in allowed:
                return {"action": "wait", "notes": f"Illegal talk rejected for {agent_name}"}
        return action

    def history(self) -> List[Dict[str, object]]:
        return list(self.debug_history)
