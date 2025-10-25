"""Sequential two-agent AutoGen sample backed by CLI model clients."""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from cli_clients import GeminiCliChatCompletionClient


async def _run_agent(agent: AssistantAgent, task) -> str:
    result = await agent.run(task=task)
    for message in reversed(result.messages):
        if hasattr(message, "to_text"):
            text = message.to_text()
        else:
            text = getattr(message, "content", "")
        if text:
            return text
    raise RuntimeError(f"{agent.name} did not return a textual message.")


async def main() -> None:
    def make_agent(name: str, system_message: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=system_message,
            model_client=GeminiCliChatCompletionClient(debug=True),
        )

    planner = make_agent(
        "planner",
        (
            "You are a structured planner. Break each assignment into at most three concise steps another agent can "
            "execute. Respond in plain text only and never attempt to invoke tools or shell commands."
        ),
    )
    executor = make_agent(
        "executor",
        (
            "You elaborate on each received plan step with helpful detail and practical advice. Stick to plain text and "
            "avoid calling tools or shell commands."
        ),
    )

    try:
        planner_text = await _run_agent(
            planner,
            "Propose a concise three-step plan for integrating Gemini CLI into an AutoGen workflow.",
        )
    except Exception as exc:
        print(f"[Planner error] {exc}")
        return

    seed_messages = [
        TextMessage(
            content="Please expand on each of the planner's steps with practical implementation detail.",
            source="user",
        ),
        TextMessage(content=planner_text, source=planner.name),
    ]

    try:
        executor_text = await _run_agent(executor, seed_messages)
    except Exception as exc:
        print(f"[Executor error] {exc}")
        return

    print("=== Planner Proposal ===")
    print(planner_text)
    print()
    print("=== Executor Expansion ===")
    print(executor_text)


if __name__ == "__main__":
    asyncio.run(main())

