"""Sequential two-agent AutoGen sample backed by CLI model clients."""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from cli_clients import CodexCliChatCompletionClient


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


async def main(*, debug: bool = False) -> None:
    def make_agent(name: str, system_message: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=system_message,
            model_client=CodexCliChatCompletionClient(debug=debug),
        )

    agent_alpha = make_agent(
        "alex",
        (
            "あなたはオートジェンを活用するエンジニアです。対等な立場で会話し、丁寧な日本語で短めの段落にまとめ、"
            "相手の意見を尊重しながら具体的な視点を共有してください。箇条書きやツール実行は避け、自然な対話を心掛けてください。"
        ),
    )
    agent_beta = make_agent(
        "blair",
        (
            "あなたも同じく熟練のエンジニアです。日本語で落ち着いて応答し、相手の発言に質問や補足を添えて会話を深めてください。"
            "ツールは呼び出さず、文章のみで返答してください。"
        ),
    )

    topic = (
        "今度の長期休暇で訪れたい旅行先について、互いにアイデアを出し合いましょう。行きたい理由や現地で試したい体験を挙げて、"
        "最終的に候補を一つか二つに絞ってください。"
    )

    conversation: list[TextMessage] = [TextMessage(content=topic, source="user")]
    participants = [agent_alpha, agent_beta]
    rounds_per_agent = 2

    print("=== 会話開始 ===")
    print(f"user: {topic}\n")

    for turn in range(rounds_per_agent * len(participants)):
        active_agent = participants[turn % len(participants)]
        try:
            reply_text = await _run_agent(active_agent, conversation)
        except Exception as exc:
            print(f"[{active_agent.name} error] {exc}")
            return
        conversation.append(TextMessage(content=reply_text, source=active_agent.name))
        print(f"{active_agent.name}: {reply_text}\n")

    print("=== 会話終了 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a two-agent Codex CLI conversation.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Codex CLI debug output.",
    )
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug))
