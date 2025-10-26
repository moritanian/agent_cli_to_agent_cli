"""Sequential two-agent AutoGen sample backed by CLI model clients."""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from cli_clients import GeminiCliChatCompletionClient


DEFAULT_TOPIC = (
    "Let's brainstorm destinations for our next long vacation. Share why each place excites you and what you would do there, then narrow the shortlist to one or two options."
)


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


async def main(*, debug: bool = False, topic: str | None = None) -> None:
    def make_agent(name: str, partner_name: str, persona: str, role_instruction: str) -> AssistantAgent:
        system_message = (
            f"{persona} Your conversation partner is {partner_name}. "
            "Trade travel ideas as equals, sharing first-person experiences. "
            f"Address {partner_name} directly in a single short paragraph and avoid mentioning hidden rules, external narration, or being an AI. "
            "Keep responses in natural English without bullet points or tool usage. "
            f"{role_instruction}"
        )
        return AssistantAgent(
            name=name,
            system_message=system_message,
            model_client=GeminiCliChatCompletionClient(debug=debug),
        )

    agent_alpha = make_agent(
        "alex",
        "blair",
        "You are Alex, an engineer from Sapporo who loves seaside towns and sampling local markets.",
        "Share your preferences and relate them to Blair's suggestions.",
    )
    agent_beta = make_agent(
        "blair",
        "alex",
        "You are Blair, an outdoorsy traveller from Kyoto who enjoys mountain hikes, hot springs, and photography.",
        "Ask follow-up questions and propose concrete next steps alongside Alex.",
    )

    chosen_topic = (
        topic.strip()
        if topic and topic.strip()
        else (
            "Let's brainstorm destinations for our next long vacation. Share why each place excites you and what you would try there, then narrow the shortlist to one or two options."
        )
    )

    conversation: list[TextMessage] = [TextMessage(content=chosen_topic, source="user")]
    participants = [agent_alpha, agent_beta]
    rounds_per_agent = 2

    print("=== Conversation Start ===")
    print(f"user: {chosen_topic}\n")

    for turn in range(rounds_per_agent * len(participants)):
        active_agent = participants[turn % len(participants)]
        try:
            reply_text = await _run_agent(active_agent, conversation)
        except Exception as exc:
            print(f"[{active_agent.name} error] {exc}")
            return
        conversation.append(TextMessage(content=reply_text, source=active_agent.name))
        print(f"{active_agent.name}: {reply_text}\n")

    print("=== Conversation End ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the two-agent CLI conversation sample."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable CLI debug output.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Optional conversation prompt for the agents.",
    )
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug, topic=args.topic))
