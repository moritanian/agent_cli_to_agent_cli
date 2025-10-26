"""Sequential two-agent AutoGen sample backed by CLI model clients."""

from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from cli_clients import GeminiCliChatCompletionClient


DEFAULT_TOPIC = (
    "今度の長期休暇で訪れたい旅行先について、互いにアイデアを出し合いましょう。"
    "行きたい理由や現地で試したい体験を挙げて、最終的に候補を一つか二つに絞ってください。"
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
            f"{persona}。会話のパートナーは{partner_name}です。"
            "与えられた話題について親しい仲間として語り合い、自分の考えや体験を一人称で共有してください。"
            f"返答は自然な会話として{partner_name}の名前を含めると親しみやすいです。"
            f"{partner_name}に直接語りかける1段落以内の日本語で返答し、AIである旨や第三者目線の解説、ユーザー向けのまとめは書かないでください。"
            "箇条書きやツール実行は避け、相手の発言に共感や質問、提案を添えて会話を前進させてください。"
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
        "あなたはエンジニアのAlexで、札幌出身。土地の食文化や海辺の街が好きで、休暇にはよく現地の市場を回ります",
        "相手の提案に共感しつつ、自分の好みや経験を素直に述べてください。",
    )
    agent_beta = make_agent(
        "blair",
        "alex",
        "あなたはアウトドア派のBlairで、京都出身。山歩きや温泉が好きで、写真撮影が趣味です",
        "相手の意見に質問や具体的な案を添えて、次のステップを一緒に考えてください。",
    )

    chosen_topic = topic.strip() if topic and topic.strip() else DEFAULT_TOPIC

    conversation: list[TextMessage] = [TextMessage(content=chosen_topic, source="user")]
    participants = [agent_alpha, agent_beta]
    rounds_per_agent = 2

    print("=== 会話開始 ===")
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

    print("=== 会話終了 ===")


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
